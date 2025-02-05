import os
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, regularizers, callbacks
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B2, preprocess_input
from sklearn.utils import class_weight
import json
import keras_tuner as kt
import datetime
import re

"""
Model Overview:
This model uses transfer learning with EfficientNetV2B2, a convolutional neural network (CNN) pre-trained on ImageNet, to classify plant diseases from leaf images.
The architecture has been enhanced with custom dense layers, dropout, and L2 regularization to prevent overfitting and improve the model’s generalization.

Training Process:

1. Dataset Processing:
   - Images are loaded from the dataset and resized to 300×300 pixels for optimal feature extraction.
   - Data augmentation is applied, including random flipping, rotation, zoom, contrast adjustments, and brightness to improve generalization.
   - The model uses datasets such as:
     - Plant Village (Kaggle)
     - Pomegranate
     - Bananas
     - Plant Seq
     - Cannabis
   - Target Disease: Cassava is included, considering the prevalence of gemini virus.

2. Feature Extraction:
   - The EfficientNetV2B2 base model is initially frozen to leverage the pre-trained features from ImageNet.
   - A custom head with dense layers, dropout, and batch normalization is added for better performance on plant disease classification.
   - The model's final fully connected layers use softmax activation to output class probabilities for each disease.

3. Optimization:
   - Adam Optimizer: An adaptive learning rate schedule is used, with an initial learning rate of 1e-3 and a decay rate of 0.96, ensuring steady convergence.
   - Loss Function: Sparse Categorical Crossentropy is used, suitable for multi-class classification.
   - Metrics: Precision, Recall, and Accuracy are used, with custom SparsePrecision and SparseRecall metrics to handle sparse categorical data effectively.

4. Class Balancing:
   - Class weights are computed to address imbalances in the dataset, ensuring the model focuses equally on all classes, particularly underrepresented plant diseases.

5. Fine-tuning:
   - After the initial training phase, the top layers of EfficientNetV2B2 are unfrozen and trained at a lower learning rate (1e-5) to fine-tune the model and further enhance classification accuracy.

6. Hyperparameter Tuning:
   - Random Search is used to fine-tune the model’s hyperparameters, including dropout rates and L2 regularization. This process, though time-consuming, ensures the best possible configuration for the model.
   - Hyperparameters: Dropout rates range from 0.2 to 0.5, and L2 regularization varies from 1e-5 to 1e-3.

Inference (Prediction Process):
- Test-Time Augmentation (TTA): During inference, the model generates multiple augmented versions of the input image to enhance prediction robustness.
- The model averages predictions from these augmented versions, improving the confidence of the class prediction.
- The most probable class label is identified, with a corresponding confidence score provided.

Predicted Output:
- The prediction includes the fruit type, disease name, and confidence level. A regular expression extracts and cleans up the predicted class label to return readable output (e.g., "Banana", "Leaf Spot").

Code Structure:
1. Data Pipeline: The _create_data_pipeline() method handles the dataset loading, preprocessing, and augmentation, ensuring the data is efficiently prepared for training.
2. Model Construction: The model is built using EfficientNetV2B2 as the base model, with additional dense layers and regularization techniques.
3. Training: The train() method manages both the initial training and fine-tuning phases. It incorporates class weights, early stopping, and model checkpointing to optimize the training process.
4. Prediction: The predict() method loads a trained model and class labels, applies preprocessing to the input image, and performs predictions with test-time augmentation for enhanced accuracy.


-- Fabian
"""

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    BATCH_SIZE = 16
    INITIAL_EPOCHS = 30
    FINE_TUNE_EPOCHS = 5
    VALIDATION_SPLIT = 0.2
    SEED = 123
    INITIAL_LR = 1e-3
    FINE_TUNE_LR = 1e-5
    DROPOUT_RATE = 0.3
    LABEL_SMOOTHING = 0.1
    TTA_STEPS = 5  # Test-time augmentation steps

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
        else:
            dropout_rate = self.DROPOUT_RATE
            l2_reg = 1e-4

        # Transfer learning base (EfficientNetV2B2)
        base_model = EfficientNetV2B2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
            pooling='avg'
        )
        base_model.trainable = False

        # Custom head
        inputs = layers.Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3))
        x = base_model(inputs)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256,
            activation='swish',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(len(self.class_labels),
            activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Custom learning rate with decay
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.INITIAL_LR,
            decay_steps=1000,
            decay_rate=0.96
        )

        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr_schedule),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=[
                'accuracy',
                SparsePrecision(name='precision'),
                SparseRecall(name='recall')
            ]
        )
        return model

    def _get_callbacks(self):
        return [
            callbacks.EarlyStopping(
                monitor='val_loss',
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

    def hyperparameter_tune(self, train_ds, val_ds):
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
                model = self.hyperparameter_tune(train_ds, val_ds)
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

            # Fine-tuning
            logger.info("Starting fine-tuning...")
            base_model = model.layers[1]
            base_model.trainable = True
            for layer in base_model.layers[:-4]:
                layer.trainable = False

            model.compile(
                optimizer=optimizers.Adam(self.FINE_TUNE_LR),
                loss=losses.SparseCategoricalCrossentropy(),
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

    def predict(self, image_path):
        try:
            # Load model and class labels
            model = keras.models.load_model(self.output_model_path)
            with open(f"{self.output_model_path}_class_names.json", 'r') as f:
                class_names = json.load(f)

            # Load and preprocess image
            img = keras.preprocessing.image.load_img(
                image_path, target_size=(self.IMG_SIZE, self.IMG_SIZE))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            # Apply preprocessing (use the correct preprocessing method)
            img_array = preprocess_input(img_array)

            # Test-time augmentation
            predictions = []
            for _ in range(self.TTA_STEPS):
                augmented = self._apply_tta_augmentation(img_array)
                predictions.append(model.predict(augmented, verbose=0))

            avg_prediction = np.mean(predictions, axis=0)
            conf = np.max(avg_prediction)
            class_idx = np.argmax(avg_prediction)
            class_label = class_names[class_idx]

            logger.info(f"Class Label: {class_label}")
            pattern = r"^([A-Za-z,]+(?:_[A-Za-z]+)*)(?:[_\(].*)? (.*)$"

            fruit, disease = re.match(pattern, class_label).groups()
            fruit = re.sub(r"_", " ", fruit)
            disease = re.sub(r"_", " ", disease).title()

            return {
                "fruit": fruit,
                "disease": disease,
                "confidence": float(conf)
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


    def _apply_tta_augmentation(self, img_array):
        # Apply input normalization first
        img_array = preprocess_input(img_array)

        # Random augmentation for TTA
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
    - Backbone: EfficientNetV2B2 (ImageNet pretrained)
    - Custom Head:
      • Global Average Pooling
      • Dense (256 units) with SWISH activation
      • Batch Normalization + Dropout
      • Softmax Classification Layer

    Key Features:
    ✅ 300x300px RGB input resolution
    ✅ Supports 50-1000 plant disease classes
    ✅ Test-Time Augmentation (TTA) for robust predictions
    ✅ EfficientNetV2B2 preprocessing for input normalization

    Purpose:
    Automated detection of plant diseases from leaf images with
    state-of-the-art deep learning for precision agriculture.
    \033[0m
    """)

    parser = argparse.ArgumentParser(description='Advanced Plant Pathogen Detection',
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=False, help='Path to training dataset')
    parser.add_argument('--output', required=True, help='Model output path')
    parser.add_argument('--image', help='Image for prediction')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
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
        detector.train(tune_hyperparams=args.tune)

if __name__ == "__main__":
    main()