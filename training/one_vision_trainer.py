import os
import logging
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import layers, models, optimizers, losses, regularizers, callbacks
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B2, preprocess_input
from tensorflow.keras.metrics import Metric
from sklearn.utils import class_weight
import json
import keras_tuner as kt
import datetime
import re
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@register_keras_serializable(package='Custom')
class ChannelAttention(layers.Layer):
    """
    Channel Attention Mechanism for enhancing feature maps based on channel-wise importance.

    This layer applies the channel attention mechanism to feature maps, allowing the model to focus
    on more informative channels, which are often more relevant to the task at hand. The mechanism
    reduces the number of channels to a smaller representation, then restores it to the original size
    through a learnable transformation.

    The attention mechanism works by computing a channel descriptor through:
    1. **Global Average Pooling**: Aggregates spatial features to obtain channel-wise statistics, which
       helps the model identify the most informative channels.
    2. **Fully connected layers**: Learnable transformations that produce attention weights for each
       channel, allowing the model to adjust the importance of each channel dynamically.
    3. **Sigmoid activation**: Applies a gating mechanism that adjusts the importance of each channel
       based on the learned weights.

    **In the context of plant pathogen detection**, the channel attention mechanism can help the model
    prioritize the most relevant channels in an image, which may correspond to specific features or
    patterns associated with plant diseases. For example, certain channels may encode visual features
    indicative of infection, such as discoloration, lesions, or abnormal textures on plant leaves. By
    focusing on these channels, the model can more effectively distinguish between healthy and infected
    plant tissues, leading to improved performance in diagnosing plant pathogens and detecting disease.

    By enabling the model to focus on key channels that hold the most relevant information, the channel
    attention mechanism improves the model's ability to classify plant pathogens based on their
    distinctive visual signatures, even in challenging cases where pathogens cause subtle or complex symptoms.
    """

    def __init__(self, reduction_ratio=16, **kwargs):
        """
        Initializes the ChannelAttention layer.

        Args:
            reduction_ratio (int): The factor by which the number of channels is reduced before applying
                                    fully connected layers (default is 16).
            **kwargs: Any additional arguments passed to the parent class (Layer).
        """
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        """
        Builds the internal layers used for channel attention.

        This method defines two fully connected layers to apply the attention mechanism.
        The first dense layer reduces the number of channels by the reduction ratio, while the
        second layer restores the channels to the original size.

        Args:
            input_shape (tuple): The shape of the input tensor.
        """
        channels = input_shape[-1]
        self.dense_1 = layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.dense_2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        """
        Performs the channel attention operation.

        This method applies the following steps:
        1. Global Average Pooling: Aggregates the spatial dimensions to form a single descriptor per channel.
        2. Passes through two fully connected layers to compute attention weights.
        3. Multiplies the input tensor with the computed attention weights to emphasize important channels.

        Args:
            inputs (tensor): The input tensor to the layer.

        Returns:
            tensor: The input tensor weighted by the attention weights for each channel.
        """
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        avg_pool = layers.Reshape((1, 1, avg_pool.shape[1]))(avg_pool)
        x = self.dense_1(avg_pool)
        x = self.dense_2(x)
        return inputs * x

    def get_config(self):
        """
        Returns the configuration of the layer, including the hyperparameters.

        This method is used for serialization, to store the layer's configuration when saving the model.

        Returns:
            dict: The configuration dictionary of the layer.
        """
        config = super(ChannelAttention, self).get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
        })
        return config

@register_keras_serializable(package='Custom')
class SpatialAttention(layers.Layer):
    """
    Spatial Attention Mechanism for enhancing feature maps based on spatial importance.

    This layer applies a spatial attention mechanism to feature maps, which allows the model
    to focus on the important spatial regions of the input. The mechanism uses a convolutional
    layer to learn the spatial attention map, which is then used to weight the input tensor.

    In the context of plant pathogen detection, the spatial attention mechanism can help the
    model focus on the critical regions of an image that are more likely to contain visual
    symptoms of a disease. These regions could include spots with discolored or malformed
    plant tissue, which are indicators of pathogen presence. By emphasizing these important
    regions, the model is better able to differentiate between healthy and infected plants.

    The spatial attention mechanism helps the model concentrate on relevant features,
    improving its ability to accurately classify or diagnose plant diseases.
    """
    def __init__(self, **kwargs):
        """
        Initializes the SpatialAttention layer.

        Args:
            **kwargs: Any additional arguments passed to the parent class (Layer).
        """
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = None

    def build(self, input_shape):
        """
        Builds the convolutional layer used for spatial attention.

        This method defines a 2D convolutional layer with a kernel size of 7x7, a stride of 1,
        and 'same' padding. The convolution is followed by a sigmoid activation to generate
        the attention weights, which will be between 0 and 1.

        Args:
            input_shape (tuple): The shape of the input tensor.
        """
        self.conv = layers.Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid')

        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        """
        Performs the spatial attention operation.

        This method generates a spatial attention map using the convolutional layer, and
        multiplies the input tensor by the generated attention map to emphasize the
        important spatial regions of the input.

        Args:
            inputs (tensor): The input tensor to the layer.

        Returns:
            tensor: The input tensor weighted by the spatial attention map.
        """
        x = self.conv(inputs)
        return inputs * x

    def get_config(self):
        """
        Returns the configuration of the layer, including the hyperparameters.

        This method is used for serialization, to store the layer's configuration when saving the model.

        Returns:
            dict: The configuration dictionary of the layer.
        """
        config = super(SpatialAttention, self).get_config()
        return config

@register_keras_serializable(package='Custom')
class HybridAttentionBlock(layers.Layer):
    """
    Hybrid Attention Block combining Channel Attention and Spatial Attention mechanisms.

    This block applies both **Channel Attention** and **Spatial Attention** to the input tensor, allowing the
    model to focus on important channels (features) as well as relevant spatial regions. These attention mechanisms
    are critical for improving the performance of the model in tasks where both feature-specific and spatial
    information are essential, such as in **plant pathogen detection** using image data.

    In the context of **plant pathogen detection**, this attention block helps the model:
    1. **Focus on Important Features**: Channel Attention allows the model to selectively weigh the importance
       of different feature maps, which is crucial when identifying specific patterns in plant images, such as
       subtle signs of disease or pathogen-related symptoms.
    2. **Highlight Relevant Regions**: Spatial Attention helps the model focus on the spatial regions in the image
       that are most indicative of a plant pathogen, such as leaf lesions, discoloration, or structural damage.

    This combination of attention mechanisms helps the model learn both **which features** and **where** in the
    image to focus, improving its ability to detect plant diseases with greater precision, especially in complex
    and noisy plant images.

    This block is part of a larger neural network architecture and can be used to improve the accuracy of
    plant disease classification, enabling early detection of various plant pathogens based on visual cues.
    """
    def __init__(self, **kwargs):
        """
        Initializes the HybridAttentionBlock with ChannelAttention and SpatialAttention sub-layers.

        Args:
            **kwargs: Additional arguments for the base Layer class.
        """
        super(HybridAttentionBlock, self).__init__(**kwargs)
        self.channel_attention = ChannelAttention()
        self.spatial_attention = SpatialAttention()

    def build(self, input_shape):
        """
        Builds the HybridAttentionBlock by initializing sub-layers.

        This ensures that both the channel and spatial attention mechanisms are properly built before
        the block is used in the forward pass.

        Args:
            input_shape (tuple): The shape of the input tensor.
        """
        # Call build methods of sub-layers to ensure they are built
        self.channel_attention.build(input_shape)
        self.spatial_attention.build(input_shape)
        super(HybridAttentionBlock, self).build(input_shape)

    def call(self, inputs):
        """
        Applies the Hybrid Attention mechanism (Channel + Spatial) to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor (e.g., feature map from a convolutional layer).

        Returns:
            tf.Tensor: The input tensor after applying both channel and spatial attention mechanisms.

        Raises:
            ValueError: If the input tensor does not have 4 dimensions (batch_size, height, width, channels).
        """
        if len(inputs.shape) != 4:
            raise ValueError(f"Expected 4D tensor, got {len(inputs.shape)}D tensor.")
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        """
        Returns the configuration of the HybridAttentionBlock, allowing for the saving and
        loading of the block with the same configuration.

        Returns:
            dict: Configuration of the layer.
        """
        config = super(HybridAttentionBlock, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a HybridAttentionBlock instance from its configuration.

        Args:
            config (dict): Configuration of the layer.

        Returns:
            HybridAttentionBlock: A new instance of the block.
        """
        return cls(**config)

@register_keras_serializable(package="Custom")
class SparseMacroF1(Metric):
    """
    Sparse Macro F1 Score for multi-class classification tasks, particularly useful for imbalanced classes.

    This metric calculates the Macro F1 score across all classes, which is the harmonic mean of precision
    and recall. The SparseMacroF1 is useful in scenarios where the classes are imbalanced, as it gives equal
    weight to the F1 scores of each class, regardless of the class distribution.

    **In the context of plant pathogen detection**, the SparseMacroF1 metric helps assess the model's
    ability to accurately detect various plant diseases (each represented as a class) while considering
    the precision and recall for each disease class equally. This is especially important when dealing with
    rare or less common diseases that might not appear as frequently in the training data.

    A high Macro F1 score indicates that the model is both precise and recall-effective across different
    plant pathogen classes, thus ensuring that the model is not biased toward the more common plant diseases
    and can correctly identify rare pathogens as well.

    The calculation of the Macro F1 score is as follows:
    1. **Precision**: The fraction of correct predictions for each class, specifically the correct identification
       of plant diseases.
    2. **Recall**: The fraction of actual occurrences of each plant pathogen correctly identified by the model.
    3. **F1-score**: The harmonic mean of precision and recall, capturing the trade-off between them for each
       class (disease).
    """

    def __init__(self, num_classes: int, name="macro_f1", **kwargs):
        """
        Initializes the SparseMacroF1 metric.

        Args:
            num_classes (int): The number of classes in the classification problem (representing plant diseases).
            name (str): Name of the metric (default is "macro_f1").
            **kwargs: Additional arguments for the base Metric class.

        Attributes:
            true_positives (tf.Variable): Stores the count of true positives per class.
            false_positives (tf.Variable): Stores the count of false positives per class.
            false_negatives (tf.Variable): Stores the count of false negatives per class.
        """
        super(SparseMacroF1, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes

        self.true_positives = self.add_weight(
            name="tp", shape=(num_classes,), initializer="zeros", dtype=tf.float32
        )
        self.false_positives = self.add_weight(
            name="fp", shape=(num_classes,), initializer="zeros", dtype=tf.float32
        )
        self.false_negatives = self.add_weight(
            name="fn", shape=(num_classes,), initializer="zeros", dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric state by computing the true positives, false positives,
        and false negatives for each class, in the context of plant pathogen detection.

        Args:
            y_true (tf.Tensor): True labels (sparse integer labels representing plant diseases).
            y_pred (tf.Tensor): Predicted logits or probabilities representing the likelihood of plant diseases.
            sample_weight (tf.Tensor, optional): Weights for each sample (not used here).

        Steps:
            1. Convert `y_true` to integer type (class indices for plant pathogens).
            2. Get the predicted class index by taking the argmax of `y_pred` (model's predicted disease class).
            3. For each class (plant pathogen), compute:
                - **True Positives (TP)**: Correctly identified plant pathogen (disease) instances.
                - **False Positives (FP)**: Instances where a plant pathogen was incorrectly predicted.
                - **False Negatives (FN)**: Instances where the plant pathogen was not detected but should have been.
            4. Update the metric's state variables.
        """
        y_true = tf.cast(y_true, tf.int32)
        preds = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        def compute_class_stats(i):
            """Computes TP, FP, and FN for a specific plant disease class `i`."""

            true_mask = tf.equal(y_true, i)
            pred_mask = tf.equal(preds, i)

            tp = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(true_mask), pred_mask), tf.float32))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, tf.logical_not(pred_mask)), tf.float32))

            return tp, fp, fn

        stats = tf.map_fn(compute_class_stats, tf.range(self.num_classes), dtype=(tf.float32, tf.float32, tf.float32))

        self.true_positives.assign_add(stats[0])
        self.false_positives.assign_add(stats[1])
        self.false_negatives.assign_add(stats[2])

    def result(self):
        """
        Computes and returns the Macro F1-score for plant pathogen detection.

        Steps:
            1. Calculate precision for each plant disease class: TP / (TP + FP + epsilon)
            2. Calculate recall for each plant disease class: TP / (TP + FN + epsilon)
            3. Compute F1-score for each class: 2 * (Precision * Recall) / (Precision + Recall + epsilon)
            4. Take the mean across all plant disease classes to get the Macro F1-score.

        Returns:
            tf.Tensor: The Macro F1-score for plant pathogen detection.
        """
        epsilon = 1e-7
        precision = self.true_positives / (self.true_positives + self.false_positives + epsilon)
        recall = self.true_positives / (self.true_positives + self.false_negatives + epsilon)
        f1_per_class = 2 * precision * recall / (precision + recall + epsilon)

        return tf.reduce_mean(f1_per_class)

    def reset_states(self):
        """
        Resets the state variables (TP, FP, FN) to zero. This is important when evaluating
        the metric across multiple batches, ensuring an accurate assessment of model performance
        on unseen plant pathogen data.
        """
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))



class SparseF1(tf.keras.metrics.Metric):
    """
    Computes the F1 score for sparse categorical labels, a critical metric for classification tasks
    such as **plant pathogen detection**.

    The F1 score is the harmonic mean of precision and recall, calculated as:
    F1 = 2 * (Precision * Recall) / (Precision + Recall + epsilon)

    In the context of **plant pathogen detection**, the F1 score is a valuable metric because:
    1. **Precision** measures how many of the predicted plant pathogen instances are actually correct. This is important to avoid false positives,
       where the model might mistakenly classify a healthy plant as infected.
    2. **Recall** measures how well the model detects all the actual plant pathogen cases. High recall ensures that the model can identify most of the infected plants,
       minimizing the number of false negatives (healthy plants wrongly identified as uninfected).

    The F1 score, as a combination of precision and recall, provides a balanced evaluation of model performance, especially when there is an imbalance between the number
    of healthy and infected plants in the dataset. This balance is crucial in **plant disease classification** tasks, where missing a small number of infected plants
    (false negatives) can have significant consequences on crop yield and plant health.

    This class extends `tf.keras.metrics.Metric` and internally uses `SparsePrecision` and `SparseRecall` to compute the required precision and recall values, ultimately
    calculating the F1 score.
    """
    def __init__(self, name='f1', **kwargs):
        """
        Initializes the SparseF1 metric.

        Args:
            name (str): The name of the metric (default is 'f1').
            **kwargs: Additional arguments for the base Metric class.
        """
        super().__init__(name=name, **kwargs)
        self.precision = SparsePrecision()
        self.recall = SparseRecall()
        self.f1 = self.add_weight(name='f1', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state by calculating precision and recall, and then computing the F1 score.

        Args:
            y_true (tf.Tensor): True labels (sparse integer labels indicating plant condition, such as 'healthy' or 'infected').
            y_pred (tf.Tensor): Predicted labels (sparse integer labels representing model's predictions).
            sample_weight (tf.Tensor, optional): Sample weights (not used in this case).

        Steps:
            1. Compute the precision and recall values for the current batch of predictions.
            2. Use precision and recall to calculate the F1 score.
        """
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

        p = self.precision.result()
        r = self.recall.result()
        self.f1.assign(2 * ((p * r) / (p + r + tf.keras.backend.epsilon())))

    def result(self):
        """
        Returns the current F1 score, which is the harmonic mean of precision and recall.

        Returns:
            tf.Tensor: The F1 score of the model, indicating the performance in detecting plant pathogens.
        """
        return self.f1

    def reset_state(self):
        """
        Resets the state of the metric. This is called at the start of each evaluation step.

        Resets precision, recall, and F1 score to zero to ensure that calculations for the next batch are correct.
        """
        self.precision.reset_state()
        self.recall.reset_state()
        self.f1.assign(0)

class TemperatureScaling(layers.Layer):
    """
    Temperature scaling layer for confidence calibration, which helps to adjust the model's
    confidence in its predictions. This is particularly useful in **plant pathogen detection**
    tasks where the model may need to be more confident or conservative in its predictions
    to avoid false negatives or false positives.

    In plant pathogen detection, accurate probability calibration is crucial as the model's
    confidence directly impacts decisions like whether a plant is infected or not. By using
    temperature scaling, we can adjust the model's output probabilities to better reflect
    the true likelihood of pathogen presence, which is critical for minimizing errors in
    classification (e.g., misclassifying a healthy plant as infected or vice versa).

    Temperature scaling achieves this by dividing the logits (the raw prediction values) by a
    learnable temperature factor. This softens the probabilities, ensuring that the model's
    confidence in predictions aligns with actual accuracy, which is crucial in highly imbalanced
    tasks like plant disease detection.
    """
    def __init__(self, initial_temp=1.0, max_temp=3.0, **kwargs):
        """
        Initializes the TemperatureScaling layer.

        Args:
            initial_temp (float): Initial temperature value. Default is 1.0.
            max_temp (float): Maximum allowable temperature value. Default is 3.0.
            **kwargs: Additional arguments for the base Layer class.
        """
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
        """
        Applies the temperature scaling to the model's inputs, which are the logits (raw predictions).
        The logits are divided by the learned temperature to adjust the confidence.

        Args:
            inputs (tf.Tensor): The logits output by the model.

        Returns:
            tf.Tensor: The logits scaled by the temperature.
        """
        return inputs / self.temperature

    def get_config(self):
        """
        Returns the configuration of the layer, including the initial temperature and max temperature.

        Returns:
            dict: The configuration of the layer.
        """
        config = super().get_config()
        config.update({
            'initial_temp': self.initial_temp,
            'max_temp': self.max_temp
        })
        return config

class SparsePrecision(tf.keras.metrics.Precision):
    """
    Computes the precision for sparse categorical labels, adapted for plant pathogen detection.

    Precision is the proportion of true positive predictions (correctly identified infected plants)
    among all positive predictions (predicted infected plants). In plant pathogen detection, precision
    is important because it reflects how many of the predicted infected plants are actually infected.
    A high precision means fewer healthy plants are misclassified as infected, which is crucial to prevent
    unnecessary treatments or interventions.

    In this implementation, we use `argmax` to handle the sparse categorical predictions, where each
    prediction corresponds to a class index (the predicted class label).
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the precision state by computing the true positives and false positives.

        Args:
            y_true (tf.Tensor): True labels (sparse integer labels).
            y_pred (tf.Tensor): Predicted logits or probabilities.
            sample_weight (tf.Tensor, optional): Weights for each sample (not used here).
        """
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight)

class SparseRecall(tf.keras.metrics.Recall):
    """
    Computes the recall for sparse categorical labels, adapted for plant pathogen detection.

    Recall is the proportion of true positive predictions (correctly identified infected plants)
    among all actual positive instances (actual infected plants). In plant pathogen detection, recall
    is important to ensure that as many infected plants as possible are identified, even if that means
    also flagging some healthy plants as infected (which can then be reviewed manually). A high recall
    reduces the risk of missing infected plants that could affect the crop.

    In this implementation, we use `argmax` to handle the sparse categorical predictions, where each
    prediction corresponds to a class index (the predicted class label).
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the recall state by computing the true positives and false negatives.

        Args:
            y_true (tf.Tensor): True labels (sparse integer labels).
            y_pred (tf.Tensor): Predicted logits or probabilities.
            sample_weight (tf.Tensor, optional): Weights for each sample (not used here).
        """
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

        model = keras.Model(inputs, scaled_logits)

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

    def hyperparameter_bayesian__tune(self, train_ds, val_ds):
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
        logger.info("Hyperparameter tuning with BayesianOptimization...")

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
        logger.info(f"Optimal dropout rate: {best_hps.get('dropout')}")
        logger.info(f"Optimal L2 regularization: {best_hps.get('l2')}")

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

        logger.info("Hyperparameter tuning with Hyperband...")

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
        logger.info(f"Optimal dropout rate: {best_hps.get('dropout')}")
        logger.info(f"Optimal L2 regularization: {best_hps.get('l2')}")

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
        logger.info("Hyperparameter tuning with RandomSearch...")

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
        logger.info(f"Optimal dropout rate: {best_hps.get('dropout')}")
        logger.info(f"Optimal L2 regularization: {best_hps.get('l2')}")

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
            logger.error(f"Training failed: {str(e)}")
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
    ✅ Custom Metrics: SparseMacroF1, SparseF1, SparsePrecision, SparseRecall
    ✅ Advanced Training Pipeline:

      ✅ Initial training with frozen backbone
      ✅ Fine-tuning with partial unfreezing
      ✅ Temperature calibration for confidence calibration
      ✅ Early stopping, learning rate scheduling, and model checkpointing

    Purpose:
    Automated detection of plant diseases from leaf images using state-of-the-art deep learning techniques for precision agriculture.
    The model incorporates advanced attention mechanisms, confidence calibration, robust data augmentation, and hyperparameter tuning for reliable predictions.
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