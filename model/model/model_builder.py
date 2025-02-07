import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, regularizers, callbacks
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B2, preprocess_input
from sklearn.utils import class_weight
import keras_tuner as kt
from .attention import ChannelAttention, SpatialAttention, HybridAttentionBlock
from .metrics import SparseMacroF1, SparseF1, SparsePrecision, SparseRecall
from .layers import TemperatureScaling
from data.data_pipeline import create_data_pipeline

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
        """Creates the data pipeline for plant pathogen classification.

            This function sets up a robust data pipeline to handle the preprocessing of plant pathogen images.
            It applies data augmentation to simulate real-world variations in plant images, such as different
            lighting conditions, rotations, zoom levels, and contrast changes. Additionally, it computes class
            weights to handle potential dataset imbalances, ensuring that rare pathogen classes are given
            appropriate importance during training.

            The pipeline:
            - Loads and splits the dataset into training and validation subsets.
            - Applies augmentation to enhance model generalization.
            - Preprocesses images to align with the model's expected input format.
            - Uses prefetching to optimize data loading performance.

            Returns:
                Tuple of (train_ds, val_ds): Processed training and validation datasets.
            """

        # Data augmentation layer
        data_augmentation = tf.keras.Sequential([
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
        """Builds and compiles the deep learning model for plant pathogen classification.

            This function constructs a convolutional neural network (CNN) using transfer learning from
            EfficientNetV2B2, a state-of-the-art architecture pre-trained on ImageNet. The model incorporates
            attention mechanisms to enhance feature extraction, fully connected layers for classification,
            and a temperature scaling layer to improve confidence calibration.

            The function supports hyperparameter tuning, allowing adjustments to:
            - Dropout rate for regularization.
            - L2 regularization strength.
            - Number of dense layers and their units.

            Key components:
            - **EfficientNetV2B2** as the backbone, with its layers frozen initially.
            - **HybridAttentionBlock** to enhance feature representation.
            - **Global Average Pooling** to reduce dimensionality.
            - **Multiple dense layers** for classification, with Swish activation and batch normalization.
            - **Temperature Scaling** for confidence calibration.
            - **Exponential Learning Rate Decay** to optimize training convergence.

            The model is compiled with:
            - **Sparse Categorical Crossentropy Loss** for multi-class classification.
            - **Adam Optimizer** with an adaptive learning rate schedule.
            - **Custom Metrics** including Sparse Precision, Recall, F1-score, and Macro F1-score.

            Args:
                hp (HyperParameters, optional): Hyperparameter tuning object from Keras Tuner.

            Returns:
                keras.Model: Compiled model ready for training.
            """

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

        model = models.Model(inputs, scaled_logits)

        # Custom learning rate with decay
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.INITIAL_LR,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )

        num_classes = len(self.class_labels)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr_schedule),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                'accuracy',
                SparsePrecision(name='precision'),
                SparseRecall(name='recall'),
                SparseF1(name='f1'),
                SparseMacroF1(num_classes=num_classes)
            ]
        )
        return model

    def _calibrate_temperature(self, model, val_ds):
        """Calibrates temperature scaling for improved confidence calibration in plant pathogen classification.

            Temperature scaling is a post-processing technique used to adjust model confidence scores, ensuring
            that predicted probabilities better reflect true likelihoods. This step is crucial for applications
            requiring reliable confidence estimates, such as disease detection in plants.

            Key process:
            - **Freezes all model layers except the TemperatureScaling layer**, ensuring only the temperature
              parameter is optimized.
            - **Recompiles the model** with a small learning rate and Sparse Categorical Crossentropy loss.
            - **Trains on the validation dataset** to fine-tune the temperature parameter, improving probability
              calibration.
            - **Uses Early Stopping** to prevent overfitting and ensure the best-calibrated temperature is retained.

            Args:
                model (keras.Model): Pre-trained model with a TemperatureScaling layer.
                val_ds (tf.data.Dataset): Validation dataset for calibration.

            Returns:
                keras.Model: Temperature-scaled model ready for inference.
            """
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

        logging.info("Calibrating temperature scaling...")
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
        self.model = model

        return model

    def _get_callbacks(self):
        """Defines training callbacks for efficient and adaptive model training.

            Callbacks are essential for optimizing model performance, preventing overfitting, and ensuring
            efficient training. This function returns a set of callbacks that:

            - **EarlyStopping**: Monitors `val_macro_f1` to stop training early if performance stagnates, preventing overfitting.
            - **ModelCheckpoint**: Saves the best model based on `val_macro_f1`, ensuring the best-performing model is retained.
            - **TensorBoard**: Enables visualization of training metrics, aiding in monitoring performance over time.
            - **ReduceLROnPlateau**: Dynamically reduces the learning rate when `val_loss` stops improving, helping the model
                converge more effectively.

            These callbacks are particularly useful for training deep learning models in plant pathogen classification,
            where careful calibration and generalization are crucial.

            Returns:
                list: A list of configured Keras callbacks.
            """
        return [
            callbacks.EarlyStopping(
                monitor='val_macro_f1',
                mode='max',
                patience=3,
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                filepath=self.output_model_path,
                monitor='val_macro_f1',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            callbacks.TensorBoard(log_dir='./logs'),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            )
        ]

    def hyperparameter_bayesian_tune(self, train_ds, val_ds):
        """Performs Bayesian Optimization for hyperparameter tuning.

            This function utilizes Keras Tuner's Bayesian Optimization to find the best hyperparameters for the model.
            Bayesian Optimization is chosen for its efficiency in navigating large search spaces with fewer trials
            compared to random search.

            - **Objective**: Maximizes `val_macro_f1`, ensuring the best performance for multi-class classification.
            - **Max Trials**: Limits to 10 trials to balance search efficiency and computational cost.
            - **Executions per Trial**: Each hyperparameter set is evaluated twice for robustness.
            - **Directory & Project Name**: Organizes tuning results under the `tuning/pathogen_detection` directory.
            - **Callback Usage**: Uses early stopping from `_get_callbacks()` to prevent unnecessary training.

            After tuning, the function logs the optimal dropout rate and L2 regularization value before
            returning the best model configuration.

            Args:
                train_ds (tf.data.Dataset): The training dataset.
                val_ds (tf.data.Dataset): The validation dataset.

            Returns:
                keras.Model: The best-tuned model based on Bayesian Optimization.
            """
        logging.info("Hyperparameter tuning with BayesianOptimization...")

        tuner = kt.BayesianOptimization(
            lambda hp: self._build_model(hp),
            objective=kt.Objective("val_macro_f1", direction="max"),
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
        logging.info(f"Optimal dropout rate: {best_hps.get('dropout')}")
        logging.info(f"Optimal L2 regularization: {best_hps.get('l2')}")

        return tuner.hypermodel.build(best_hps)

    def hyperparameter_hyperband_tune(self, train_ds, val_ds):
        """Performs Hyperband optimization for hyperparameter tuning.

            This function applies Keras Tuner's Hyperband algorithm to efficiently search for the best
            hyperparameters. Hyperband dynamically allocates resources, stopping underperforming trials early
            to prioritize promising configurations.

            - **Objective**: Maximizes `val_macro_f1` for multi-class classification performance.
            - **Max Epochs**: Limits training to 10 full epochs per configuration.
            - **Factor**: Uses a reduction factor of 3 to control resource allocation.
            - **Executions per Trial**: Each hyperparameter configuration is evaluated twice for stability.
            - **Directory & Project Name**: Saves tuning results under `tuning/pathogen_detection`.
            - **Callback Usage**: Uses early stopping from `_get_callbacks()` to prevent overfitting.

            After tuning, the function logs the best dropout rate and L2 regularization value before
            returning the optimized model.

            Args:
                train_ds (tf.data.Dataset): The training dataset.
                val_ds (tf.data.Dataset): The validation dataset.

            Returns:
                keras.Model: The best-tuned model based on Hyperband optimization.
            """

        logging.info("Hyperparameter tuning with Hyperband...")

        tuner = kt.Hyperband(
            lambda hp: self._build_model(hp),
            objective=kt.Objective("val_macro_f1", direction="max"),
            max_epochs=10,
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
        logging.info(f"Optimal dropout rate: {best_hps.get('dropout')}")
        logging.info(f"Optimal L2 regularization: {best_hps.get('l2')}")

        return tuner.hypermodel.build(best_hps)

    def hyperparameter_tune(self, train_ds, val_ds):
        """Performs hyperparameter tuning using Random Search.

            This function applies Keras Tuner's Random Search strategy to explore different hyperparameter
            configurations by randomly sampling from the defined search space.

            - **Objective**: Maximizes `val_macro_f1` for multi-class classification.
            - **Max Trials**: Runs 10 different hyperparameter configurations.
            - **Executions per Trial**: Each configuration is evaluated twice for stability.
            - **Directory & Project Name**: Saves tuning results under `tuning/pathogen_detection`.
            - **Callback Usage**: Uses early stopping from `_get_callbacks()` to prevent overfitting.

            After tuning, the function logs the best dropout rate and L2 regularization value before
            returning the optimized model.

            Args:
                train_ds (tf.data.Dataset): The training dataset.
                val_ds (tf.data.Dataset): The validation dataset.

            Returns:
                keras.Model: The best-tuned model based on Random Search optimization.
            """
        logging.info("Hyperparameter tuning with RandomSearch...")

        tuner = kt.RandomSearch(
            lambda hp: self._build_model(hp),
            objective=kt.Objective("val_macro_f1", direction="max"),
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
        logging.info(f"Optimal dropout rate: {best_hps.get('dropout')}")
        logging.info(f"Optimal L2 regularization: {best_hps.get('l2')}")

        return tuner.hypermodel.build(best_hps)

    def train(self, tune_hyperparams=False):
        """Trains the model with optional hyperparameter tuning and fine-tuning.

            This method orchestrates the entire training pipeline, including:
            - **Data Preparation**: Loads training and validation datasets via `_create_data_pipeline()`.
            - **Hyperparameter Tuning** (if enabled):
                - Supports `"random"`, `"bayesian"`, and `"hyperband"` tuning strategies.
                - Calls the appropriate tuning method based on the provided argument.
            - **Initial Training**:
                - Trains the model using predefined class weights and callbacks.
            - **Temperature Calibration**:
                - Calibrates the temperature scaling layer using `_calibrate_temperature()`.
            - **Fine-Tuning**:
                - Unfreezes the base model's last four layers while keeping earlier layers frozen.
                - Recompiles the model with a lower learning rate (`FINE_TUNE_LR`) for refinement.
            - **Final Training**:
                - Continues training for additional epochs, starting from where initial training stopped.
            - **Model Saving**:
                - Saves the trained model along with class label mappings in JSON format.

            Args:
                tune_hyperparams (bool | str, optional):
                    - If `False`, trains without tuning.
                    - If `"random"`, `"bayesian"`, or `"hyperband"`, applies respective tuning method.

            Returns:
                keras.Model: The trained model.

            Raises:
                Exception: Logs and re-raises any training failure.
            """
        try:
            train_ds, val_ds = self._create_data_pipeline()

            if tune_hyperparams:
                if tune_hyperparams == "random":
                    model = self.hyperparameter_hyperband_tune(train_ds, val_ds)
                elif tune_hyperparams == "bayesian":
                    model = self.hyperparameter_bayesian_tune(train_ds, val_ds)
                elif tune_hyperparams == "hyperband":
                    model = self.hyperparameter_hyperband_tune(train_ds, val_ds)
                else:
                    raise ExceptionType("Unsupported hyperparameter")
            else:
                model = self._build_model()

            # Initial training
            logging.info("Starting initial training...")
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
            logging.info("Starting fine-tuning...")
            base_model = model.layers[1]
            base_model.trainable = True
            for layer in base_model.layers[:-4]:
                layer.trainable = False

            model.compile(
                optimizer=optimizers.Adam(self.FINE_TUNE_LR),
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[
                    'accuracy',
                    SparsePrecision(name='precision'),
                    SparseRecall(name='recall'),
                    SparseF1(name='f1')
                ]
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
            logging.error(f"Training failed: {str(e)}")
            raise

    def evaluate_calibration(self, dataset):
        """Evaluate model calibration using Expected Calibration Error (ECE).

            Expected Calibration Error (ECE) measures the discrepancy between
            a model's confidence and its actual accuracy. A well-calibrated
            model has confidence levels that closely match observed accuracies.

            Steps:
            1. **Predict Class Probabilities**: Compute logits and apply softmax.
            2. **Retrieve Ground Truth Labels**: Extract labels from the dataset.
            3. **Bin Predictions by Confidence**:
                - Use 10 bins (edges from 0 to 1) to group predictions by max probability.
            4. **Compute Per-Bin Accuracy & Confidence**:
                - For each bin, compute the accuracy and mean confidence.
                - Calculate the weighted difference (|accuracy - confidence|) per bin.
            5. **Return Final ECE Score**: Normalize by the total number of samples.

            Args:
                dataset (tf.data.Dataset): The dataset containing input samples and labels.

            Returns:
                float: Expected Calibration Error (ECE), where lower is better.
            """
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
        """
            Predict the class and confidence of an image using the trained model.

            This method loads the trained model and the corresponding class names, processes
            the input image, performs test-time augmentation (TTA) to reduce prediction variance,
            and outputs the predicted class label, associated disease, and confidence level.

            Args:
                image_path (str): The path to the image to be predicted.

            Returns:
                dict: A dictionary containing the predicted fruit, disease, and prediction confidence.
            """
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
            logging.info(f"Class Label: {class_label}")
            pattern = r"^([A-Za-z,]+(?:_[A-Za-z]+)*)(?:[_\(].*)? (.*)$"
            fruit, disease = re.match(pattern, class_label).groups()

            return {
                "fruit": re.sub(r"_", " ", fruit),
                "disease": re.sub(r"_", " ", disease).title(),
                "confidence": float(conf)
            }

        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def _apply_tta_augmentation(self, img_array):
        """
            Apply random augmentations to an image for Test-Time Augmentation (TTA).

            This function applies a series of random transformations to the input image
            to increase the model's robustness during prediction. The augmentations help
            by generating slightly altered versions of the input image to reduce
            overfitting and improve generalization. The augmentations include random
            horizontal and vertical flips, random brightness and contrast adjustments.

            Args:
                img_array (tensor): The image to be augmented, expected shape (height, width, channels).

            Returns:
                tensor: The augmented image after applying random transformations.
            """
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
