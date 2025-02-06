import os
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import layers, models, optimizers, losses, regularizers, callbacks
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B2, preprocess_input
from sklearn.utils import class_weight
import json
import keras_tuner as kt
import datetime
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

@register_keras_serializable(package='Custom')
class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.dense_1 = layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.dense_2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        avg_pool = layers.Reshape((1, 1, avg_pool.shape[1]))(avg_pool)
        x = self.dense_1(avg_pool)
        x = self.dense_2(x)
        return inputs * x

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
        })
        return config

@register_keras_serializable(package='Custom')
class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = None

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid')

        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        return inputs * x

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        return config

@register_keras_serializable(package='Custom')
class HybridAttentionBlock(layers.Layer):
    def __init__(self, **kwargs):
        super(HybridAttentionBlock, self).__init__(**kwargs)
        self.channel_attention = ChannelAttention()
        self.spatial_attention = SpatialAttention()

    def build(self, input_shape):
        # Call build methods of sub-layers to ensure they are built
        self.channel_attention.build(input_shape)
        self.spatial_attention.build(input_shape)
        super(HybridAttentionBlock, self).build(input_shape)

    def call(self, inputs):
        if len(inputs.shape) != 4:
            raise ValueError(f"Expected 4D tensor, got {len(inputs.shape)}D tensor.")
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super(HybridAttentionBlock, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class SparseF1(tf.keras.metrics.Metric):
    """Computes F1 score for sparse categorical labels"""
    def __init__(self, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = SparsePrecision()
        self.recall = SparseRecall()
        self.f1 = self.add_weight(name='f1', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

        p = self.precision.result()
        r = self.recall.result()
        self.f1.assign(2 * ((p * r) / (p + r + tf.keras.backend.epsilon())))

    def result(self):
        return self.f1

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        self.f1.assign(0)

class TemperatureScaling(layers.Layer):
    """Temperature scaling layer for confidence calibration"""
    def __init__(self, initial_temp=1.0, max_temp=3.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_temp = initial_temp
        self.max_temp = max_temp
        self.temperature = self.add_weight(
            name='temperature',
            shape=(1,),
            initializer=tf.constant_initializer(initial_temp),
            constraint=tf.keras.constraints.MinMaxNorm(
                min_value=0.1, max_value=max_temp
            ),
            trainable=True
        )

    def call(self, inputs):
        return inputs / self.temperature

    def get_config(self):
        config = super().get_config()
        config.update({
            'initial_temp': self.initial_temp,
            'max_temp': self.max_temp
        })
        return config

class SparsePrecision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight)

class SparseRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight)

class AdvancedPathogenDetector:
    # Configuration
    IMG_SIZE = 300  # Increased for better feature capture
    BATCH_SIZE = 32
    INITIAL_EPOCHS = 30
    FINE_TUNE_EPOCHS = 5
    VALIDATION_SPLIT = 0.2
    SEED = 123
    INITIAL_LR = 1e-3
    FINE_TUNE_LR = 1e-5
    DROPOUT_RATE = 0.3
    LABEL_SMOOTHING = 0.1
    TTA_STEPS = 5  # Test-time augmentation steps
    CALIBRATION_EPOCHS = 5
    CALIBRATION_LR = 1e-4

    def __init__(self, dataset_path, output_model_path):
        self.dataset_path = dataset_path
        self.output_model_path = output_model_path
        self.class_labels = None
        self.class_weights = None
        self._validate_paths()

    def _validate_paths(self):
        # Only validate dataset path if we're in training mode
        if self.dataset_path and not os.path.isdir(self.dataset_path):
            raise ValueError(f"Dataset directory not found: {self.dataset_path}")

        # Always validate output path
        output_dir = os.path.dirname(self.output_model_path)
        if output_dir:  # Only create if path contains directories
            os.makedirs(output_dir, exist_ok=True)

    def _create_data_pipeline(self):
        # Data augmentation layer
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ])

        # Load datasets
        original_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.dataset_path,
            validation_split=self.VALIDATION_SPLIT,
            subset="training",
            seed=self.SEED,
            image_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            label_mode='int'
        )

        original_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.dataset_path,
            validation_split=self.VALIDATION_SPLIT,
            subset="validation",
            seed=self.SEED,
            image_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            label_mode='int'
        )

        # Save class labels from original dataset
        self.class_labels = original_train_ds.class_names

        # Calculate class weights
        train_labels = np.concatenate([y for x, y in original_train_ds], axis=0)
        self.class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        self.class_weights = dict(enumerate(self.class_weights))

        # Apply augmentation and prefetch to training data
        train_ds = original_train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            lambda x, y: (preprocess_input(x), y),  # Apply preprocessing here
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(buffer_size=tf.data.AUTOTUNE)

        # Prefetch validation data and apply preprocessing
        val_ds = original_val_ds.map(
            lambda x, y: (preprocess_input(x), y),  # Apply preprocessing here
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, val_ds

    def _build_model(self, hp=None):
        # Hyperparameter tuning support
        if hp:
            dropout_rate = hp.Float('dropout', 0.2, 0.5, step=0.1)
            l2_reg = hp.Float('l2', 1e-5, 1e-3, sampling='log')
            num_dense_layers = hp.Int('num_dense_layers', 1, 3)  # Add for tuning
            dense_units = hp.Int('dense_units', 128, 512, step=64)
        else:
            dropout_rate = self.DROPOUT_RATE
            l2_reg = 1e-4
            num_dense_layers = 2  # Default number of layers
            dense_units = 256     # Default units

        # Transfer learning base (EfficientNetV2B2)
        base_model = EfficientNetV2B2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
            pooling=None
        )
        base_model.trainable = False

        inputs = layers.Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))

        x = base_model(inputs)

        x = HybridAttentionBlock()(x)
        x = layers.GlobalAveragePooling2D()(x)

        for _ in range(num_dense_layers):  # Adding multiple dense layers
            x = layers.Dense(dense_units, activation='swish', kernel_regularizer=regularizers.l2(l2_reg))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)

        logits = layers.Dense(len(self.class_labels))(x)
        temp_scaler = TemperatureScaling()
        temp_scaler.trainable = False  # Freeze temperature during initial training
        scaled_logits = temp_scaler(logits)

        model = keras.Model(inputs, scaled_logits)

        # Custom learning rate with decay
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.INITIAL_LR,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )

        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr_schedule),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                'accuracy',
                SparsePrecision(name='precision'),
                SparseRecall(name='recall'),
                SparseF1(name='f1')
            ]
        )
        return model

    def _calibrate_temperature(self, model, val_ds):
            """Calibrate temperature scaling on validation set"""
            # Freeze all layers except temperature scaling
            for layer in model.layers:
                if isinstance(layer, TemperatureScaling):
                    layer.trainable = True
                else:
                    layer.trainable = False

            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.CALIBRATION_LR),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

            logger.info("Calibrating temperature scaling...")
            history = model.fit(
                val_ds,
                epochs=self.CALIBRATION_EPOCHS,
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor='loss',
                        patience=2,
                        restore_best_weights=True
                    )
                ]
            )
            return model

    def _get_callbacks(self):
        return [
            callbacks.EarlyStopping(
                monitor='loss',
                patience=3,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                filepath=self.output_model_path,
                save_best_only=True,
                monitor='val_accuracy'
            ),
            callbacks.TensorBoard(log_dir='./logs')
        ]

    def hyperparameter_bayesian__tune(self, train_ds, val_ds):
        logger.info("Hyperparameter tuning with BayesianOptimization...")

        tuner = kt.BayesianOptimization(
            lambda hp: self._build_model(hp),
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=2,
            directory='tuning',
            project_name='pathogen_detection',
            overwrite=True
        )

        tuner.search(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=[self._get_callbacks()[0]]
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Optimal dropout rate: {best_hps.get('dropout')}")
        logger.info(f"Optimal L2 regularization: {best_hps.get('l2')}")

        return tuner.hypermodel.build(best_hps)

    def hyperparameter_hyperband_tune(self, train_ds, val_ds):
        logger.info("Hyperparameter tuning with Hyperband...")

        tuner = kt.Hyperband(
            lambda hp: self._build_model(hp),
            objective='val_accuracy',
            max_epochs=50,
            factor=3,
            executions_per_trial=2,
            directory='tuning',
            project_name='pathogen_detection',
            overwrite=True
        )

        tuner.search(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[self._get_callbacks()[0]]
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Optimal dropout rate: {best_hps.get('dropout')}")
        logger.info(f"Optimal L2 regularization: {best_hps.get('l2')}")

        return tuner.hypermodel.build(best_hps)

    def hyperparameter_tune(self, train_ds, val_ds):
        logger.info("Hyperparameter tuning with RandomSearch...")

        tuner = kt.RandomSearch(
            lambda hp: self._build_model(hp),
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=2,
            directory='tuning',
            project_name='pathogen_detection'
        )

        tuner.search(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=[self._get_callbacks()[0]]
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Optimal dropout rate: {best_hps.get('dropout')}")
        logger.info(f"Optimal L2 regularization: {best_hps.get('l2')}")

        return tuner.hypermodel.build(best_hps)

    def train(self, tune_hyperparams=False):
        try:
            train_ds, val_ds = self._create_data_pipeline()
            
            if tune_hyperparams:
                if tune_hyperparams=="random":
                  model = self.hyperparameter_hyperband_tune(train_ds, val_ds)
                elif tune_hyperparams == "bayesian":
                   model = self.hyperparameter_bayesian__tune(train_ds, val_ds)
                elif tune_hyperparams == "hyperband":
                   model = self.hyperparameter_hyperband_tune(train_ds, val_ds)
                else:
                    raise ExceptionType("Unsupported hyperparameter")
            else:
                model = self._build_model()

            # Initial training
            logger.info("Starting initial training...")
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.INITIAL_EPOCHS,
                class_weight=self.class_weights,
                callbacks=self._get_callbacks()
            )

            # Temperature calibration
            model = self._calibrate_temperature(model, val_ds)

            # Fine-tuning
            logger.info("Starting fine-tuning...")
            base_model = model.layers[1]
            base_model.trainable = True
            for layer in base_model.layers[:-4]:
                layer.trainable = False

            model.compile(
                optimizer=optimizers.Adam(self.FINE_TUNE_LR),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.INITIAL_EPOCHS + self.FINE_TUNE_EPOCHS,
                initial_epoch=history.epoch[-1],
                class_weight=self.class_weights,
                callbacks=self._get_callbacks()
            )

            # Save final model
            model.save(self.output_model_path)
            with open(f"{self.output_model_path}_class_names.json", 'w') as f:
                json.dump(self.class_labels, f)

            return model

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def evaluate_calibration(self, dataset):
            """Evaluate model calibration using Expected Calibration Error"""
            logits = self.model.predict(dataset)
            probs = tf.nn.softmax(logits)
            labels = np.concatenate([y for x, y in dataset], axis=0)

            # Bin calculation
            bin_edges = np.linspace(0, 1, 11)
            bin_indices = np.digitize(np.max(probs, axis=1), bin_edges)

            ece = 0.0
            for b in range(1, 11):
                mask = bin_indices == b
                if np.sum(mask) > 0:
                    bin_probs = probs[mask]
                    bin_labels = labels[mask]
                    bin_acc = np.mean(np.argmax(bin_probs, axis=1) == bin_labels)
                    bin_conf = np.mean(np.max(bin_probs, axis=1))
                    ece += np.abs(bin_acc - bin_conf) * np.sum(mask)

            return ece / len(labels)

    def predict(self, image_path):
        try:
            # Load model with custom objects and class labels
            model = keras.models.load_model(
                self.output_model_path,
                custom_objects={
                    'TemperatureScaling': TemperatureScaling,
                    'SparseF1': SparseF1,
                    'SparsePrecision': SparsePrecision,
                    'SparseRecall': SparseRecall,
                    'HybridAttentionBlock': HybridAttentionBlock,
                    'ChannelAttention': ChannelAttention,
                    'SpatialAttention': SpatialAttention
                }
            )
            with open(f"{self.output_model_path}_class_names.json", 'r') as f:
                class_names = json.load(f)

            # Load and preprocess image
            img = keras.preprocessing.image.load_img(
                image_path, target_size=(self.IMG_SIZE, self.IMG_SIZE))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Shape: (1, 300, 300, 3)

            # Test-time augmentation with temperature scaling
            logits = np.zeros((1, len(class_names)))
            for _ in range(self.TTA_STEPS):
                augmented = self._apply_tta_augmentation(img_array)
                preprocessed = preprocess_input(augmented)
                logits += model.predict(preprocessed, verbose=0)

            # Average logits and apply final softmax
            avg_logits = logits / self.TTA_STEPS
            probabilities = tf.nn.softmax(avg_logits).numpy()

            # Get prediction results
            conf = np.max(probabilities)
            class_idx = np.argmax(probabilities)
            class_label = class_names[class_idx]

            # Clean up label formatting
            logger.info(f"Class Label: {class_label}")
            pattern = r"^([A-Za-z,]+(?:_[A-Za-z]+)*)(?:[_\(].*)? (.*)$"
            fruit, disease = re.match(pattern, class_label).groups()

            return {
                "fruit": re.sub(r"_", " ", fruit),
                "disease": re.sub(r"_", " ", disease).title(),
                "confidence": float(conf)
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


    def _apply_tta_augmentation(self, img_array):
        # Random augmentation for TTA (without preprocessing)
        return tf.image.random_flip_left_right(
            tf.image.random_brightness(
                tf.image.random_contrast(
                    tf.image.random_flip_up_down(img_array),
                    lower=0.8, upper=1.2
                ),
                max_delta=0.1
            )
        )


def main():
    print(r"""
          _|_|                        _|      _|  _|            _|
        _|    _|  _|_|_|      _|_|    _|      _|        _|_|_|        _|_|    _|_|_|
        _|    _|  _|    _|  _|_|_|_|  _|      _|  _|  _|_|      _|  _|    _|  _|    _|
        _|    _|  _|    _|  _|          _|  _|    _|      _|_|  _|  _|    _|  _|    _|
          _|_|    _|    _|    _|_|_|      _|      _|  _|_|_|    _|    _|_|    _|    _|
        """)

    print(f"""\033[92m
    🌱 OneVision - Advanced Plant Pathogen Detection System 🌿
    Copyright (c) {datetime.datetime.now().year} Fabian Franco-Roldan

    Model Architecture:

    Backbone: EfficientNetV2B2 (ImageNet pretrained)

    Custom Head:
    • Hybrid Attention Block (Channel + Spatial Attention)
    • Global Average Pooling
    • Dense Layers (256 units) with SWISH activation
    • Batch Normalization + Dropout
    • Temperature Scaling for confidence calibration
    • Softmax Classification Layer

    Key Features:
    ✅ 300x300px RGB input resolution
    ✅ Supports 50-1000 plant disease classes
    ✅ Test-Time Augmentation (TTA) for robust predictions
    ✅ Hybrid Attention Mechanism for enhanced feature extraction
    ✅ Temperature Scaling for calibrated confidence scores
    ✅ EfficientNetV2B2 preprocessing for input normalization
    ✅ Class Weighting for imbalanced datasets
    ✅ Hyperparameter Tuning (Random Search, Bayesian Optimization, Hyperband)

    Purpose:
    Automated detection of plant diseases from leaf images using state-of-the-art deep learning techniques for precision agriculture.
    The model incorporates advanced attention mechanisms, confidence calibration, and robust data augmentation for reliable predictions.
    \033[0m
    """)

    parser = argparse.ArgumentParser(description='Advanced Plant Pathogen Detection',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=False, help='Path to training dataset')
    parser.add_argument('--output', required=True, help='Model output path')
    parser.add_argument('--image', help='Image for prediction')
    parser.add_argument('--tune', type=str, choices=['random', 'bayesian', 'hyperband'],
                        help='Enable hyperparameter tuning with the specified method (random, bayesian, or hyperband)')
    args = parser.parse_args()

    # Validate arguments
    if not args.image and not args.dataset:
        parser.error("You must specify either --dataset for training or --image for prediction")
    if args.image and args.dataset:
        parser.error("Cannot specify both --dataset and --image")

    detector = AdvancedPathogenDetector(args.dataset, args.output)

    if args.image:
        result = detector.predict(args.image)
        print(json.dumps(result, indent=2))
    elif args.dataset:
        if args.tune:
            if args.tune == 'random':
                detector.train(tune_hyperparams='random')
            elif args.tune == 'bayesian':
                detector.train(tune_hyperparams='bayesian')
            elif args.tune == 'hyperband':
                detector.train(tune_hyperparams='hyperband')
        else:
            detector.train(tune_hyperparams=False)

if __name__ == "__main__":
    main()