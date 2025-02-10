from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.efficientnet_v2 import preprocess_input
import json
import io
import logging
import re
from PIL import Image, ImageFilter, ImageOps
import base64
import datetime
from tensorflow.keras.utils import register_keras_serializable
from sklearn.utils import class_weight
from tensorflow.keras.metrics import Metric
from tensorflow.keras import layers, models
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@register_keras_serializable(package='Custom')
class SeparableBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding="same", use_bn=True, activation="swish", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bn = use_bn
        self.activation = activation

        # Depthwise convolution captures spatial relationships without increasing parameter count.
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=1,
            use_bias=False  # No bias needed since batch normalization is applied
        )

        # Batch normalization stabilizes training and prevents internal covariate shift.
        self.bn1 = layers.BatchNormalization() if use_bn else None

        # Pointwise convolution projects the depthwise-convolved features into the desired number of filters.
        self.pointwise_conv = layers.Conv2D(filters, 1, use_bias=False)

        # Second batch normalization step.
        self.bn2 = layers.BatchNormalization() if use_bn else None

        # Activation function introduces non-linearity; "swish" is effective for feature-rich datasets.
        self.act = layers.Activation(activation) if activation else None

    def build(self, input_shape):

        # Build depthwise convolution layer
        self.depthwise_conv.build(input_shape)
        depthwise_output_shape = self.depthwise_conv.compute_output_shape(input_shape)

        # Batch normalization for depthwise convolution output
        if self.bn1 is not None:
            self.bn1.build(depthwise_output_shape)

        # Build pointwise convolution
        self.pointwise_conv.build(depthwise_output_shape)
        pointwise_output_shape = self.pointwise_conv.compute_output_shape(depthwise_output_shape)

        # Batch normalization for pointwise convolution output
        if self.bn2 is not None:
            self.bn2.build(pointwise_output_shape)

        # Activation function setup
        if self.act is not None:
            self.act.build(pointwise_output_shape)

        super(SeparableBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        if self.bn1:
            x = self.bn1(x)
        x = self.pointwise_conv(x)
        if self.bn2:
            x = self.bn2(x)
        return self.act(x) if self.act else x

    def compute_output_shape(self, input_shape):
        depthwise_shape = self.depthwise_conv.compute_output_shape(input_shape)
        pointwise_shape = self.pointwise_conv.compute_output_shape(depthwise_shape)
        return pointwise_shape

    def get_config(self):
        config = super(SeparableBlock, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bn": self.use_bn,
            "activation": self.activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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
@register_keras_serializable(package='Custom')
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
@register_keras_serializable(package='Custom')
class SparseMacroF1(Metric):
    def __init__(self, num_classes: int, name="macro_f1", **kwargs):
        super(SparseMacroF1, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes

        # Initializing tensors to store the count of true positives (tp), false positives (fp),
        # and false negatives (fn) for each class (pathogen) in the classification task.
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
        # Casting true labels to integers for comparison
        y_true = tf.cast(y_true, tf.int32)
        # Getting the predicted class (pathogen) by selecting the index with the highest probability
        preds = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

        def compute_class_stats(i):
            true_mask = tf.equal(y_true, i)  # Mask for actual class i
            pred_mask = tf.equal(preds, i)  # Mask for predicted class i

            tp = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32))  # True positives
            fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(true_mask), pred_mask), tf.float32))  # False positives
            fn = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, tf.logical_not(pred_mask)), tf.float32))  # False negatives

            return tp, fp, fn

        # Computing stats (tp, fp, fn) for all classes (pathogens)
        stats = tf.map_fn(compute_class_stats, tf.range(self.num_classes), dtype=(tf.float32, tf.float32, tf.float32))

        # Updating accumulated true positives, false positives, and false negatives
        self.true_positives.assign_add(stats[0])
        self.false_positives.assign_add(stats[1])
        self.false_negatives.assign_add(stats[2])

    def result(self):
        epsilon = 1e-7  # Small constant to avoid division by zero

        # Calculate precision and recall for each class (pathogen)
        precision = self.true_positives / (self.true_positives + self.false_positives + epsilon)
        recall = self.true_positives / (self.true_positives + self.false_negatives + epsilon)

        # Calculate F1 score per class
        f1_per_class = 2 * precision * recall / (precision + recall + epsilon)

        # Return the average macro F1 score
        return tf.reduce_mean(f1_per_class)

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

    def get_config(self):
        config = super(SparseMacroF1, self).get_config()
        config.update({"num_classes": self.num_classes})  # Add number of classes to config
        return config

    @classmethod
    def from_config(cls, config):
        # Ensure num_classes is correctly passed during deserialization
        num_classes = config.pop('num_classes')
        return cls(num_classes=num_classes, **config)

# Custom metric classes required for loading the model
@register_keras_serializable(package='Custom')
class SparsePrecision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight)
@register_keras_serializable(package='Custom')
class SparseRecall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        super().update_state(y_true, y_pred, sample_weight)
@register_keras_serializable(package="Custom")
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

class PathogenPredictor:
    IMG_SIZE = 300
    TTA_STEPS = 5  # Test-time augmentation steps

    BASE_COST = 1.0
    LETTERBOX_COST = 0.5
    COST_PER_TTA_STEP = 0.2

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_labels = None

    def load_model(self):
        """Load the trained model and class labels"""
        try:
            # Load model with custom metrics
            self.model = keras.models.load_model(
                self.model_path,
                custom_objects={
                    'SparsePrecision': SparsePrecision,
                    'SparseRecall': SparseRecall,
	                'SparseMacroF1': SparseMacroF1,
	                'SeparableBlock': SeparableBlock,
	                'HybridAttentionBlock':HybridAttentionBlock,
	                'TemperatureScaling':TemperatureScaling,
	                'SparseF1': SparseF1
                }
            )

            # Load class labels from associated JSON file
            with open(f"{self.model_path}_class_names.json", 'r') as f:
                self.class_labels = json.load(f)

            print(self.model.summary())
            logger.info("Model and labels loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, image_data):
        """Make prediction on image bytes"""
        try:
            # Check if model is loaded
            if self.model is None or self.class_labels is None:
                raise ValueError("Model not loaded - call load_model() first")

            processing_metadata = {
                'letterboxed': False,
                'tta_steps': self.TTA_STEPS
            }

            # Load image from bytes
            img = Image.open(io.BytesIO(image_data))
            img = ImageOps.exif_transpose(img)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Smart image scaling with aspect ratio preservation
            aspect_ratio = img.width / img.height
            if aspect_ratio > 2 or aspect_ratio < 0.5:
                logger.info("Image has extreme ratios - letterboxing")
                processing_metadata['letterboxed'] = True
                img.thumbnail((self.IMG_SIZE*2, self.IMG_SIZE*2))
                background = Image.new('RGB', (self.IMG_SIZE, self.IMG_SIZE), (0,0,0))
                offset = (
                    (self.IMG_SIZE - img.width) // 2,
                    (self.IMG_SIZE - img.height) // 2
                )
                background.paste(img, offset)
                img = background
            else:
                logger.info("Scaling image proportionally...")
                if max(img.size) > self.IMG_SIZE * 4:
                    img = img.resize(
                        (self.IMG_SIZE*4, self.IMG_SIZE*4),
                        resample=Image.Resampling.LANCZOS
                    )
                    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))

                img = img.resize(
                    (self.IMG_SIZE, self.IMG_SIZE),
                    resample=Image.Resampling.BILINEAR
                )

            # Convert to array and preprocess
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Shape: (1, 300, 300, 3)
            # Convert the processed image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Test-time augmentation with temperature scaling
            logits = np.zeros((1, len(self.class_labels)))
            for _ in range(self.TTA_STEPS):
                augmented = self._apply_tta_augmentation(img_array)
                preprocessed = preprocess_input(augmented)
                logits += self.model.predict(preprocessed, verbose=0)

            # Average logits and apply final softmax
            avg_logits = logits / self.TTA_STEPS
            probabilities = tf.nn.softmax(avg_logits).numpy()

            # Get prediction results
            conf = np.max(probabilities)
            class_idx = np.argmax(probabilities)
            class_label = self.class_labels[class_idx]

            # Clean up label formatting
            logger.info(f"Class Label: {class_label}")
            pattern = r"^([A-Za-z,]+(?:_[A-Za-z]+)*)(?:[_\(].*)? (.*)$"
            fruit, disease = re.match(pattern, class_label).groups()
#             img_array = tf.expand_dims(img_array, 0)
#             img_array = preprocess_input(img_array)
#
#             # Convert the processed image to base64
#             buffered = io.BytesIO()
#             img.save(buffered, format="JPEG")
#             img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
#
#             # Test-time augmentation
#             predictions = []
#             for _ in range(self.TTA_STEPS):
#                 augmented = self._apply_tta_augmentation(img_array)
#                 predictions.append(self.model.predict(augmented, verbose=0))
#
#             # Process predictions
#             avg_prediction = np.mean(predictions, axis=0)
#             conf = np.max(avg_prediction)
#             class_idx = np.argmax(avg_prediction)
#             class_label = self.class_labels[class_idx]
#
#             # Parse class label
#             pattern = r"^([A-Za-z,]+(?:_[A-Za-z]+)*)(?:[_\(].*)? (.*)$"
#             plant, disease = re.match(pattern, class_label).groups()
#             plant = re.sub(r"_", " ", plant)
#             disease = re.sub(r"_", " ", disease).title()

            credits_used = self.BASE_COST
            if processing_metadata['letterboxed']:
                credits_used += self.LETTERBOX_COST
            credits_used += self.TTA_STEPS * self.COST_PER_TTA_STEP

            return {
                "plant": fruit,
                "disease": disease,
                "confidence": float(conf),
                "processed_image": img_base64,
                "credits_used": round(credits_used, 2),
                "cost_breakdown": {
                  "base_processing": self.BASE_COST,
                  "letterboxing": self.LETTERBOX_COST if processing_metadata['letterboxed'] else 0,
                  "tta_steps": self.TTA_STEPS * self.COST_PER_TTA_STEP
                }
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def _apply_tta_augmentation(self, img_array):
        """Apply random augmentations for test-time augmentation"""
        return tf.image.random_brightness(
            tf.image.random_contrast(img_array, lower=0.8, upper=1.2),
            max_delta=0.1
        )

# Initialize FastAPI app and predictor
app = FastAPI()
predictor = PathogenPredictor(model_path="./V2B2_SB_HYB_512/model.keras")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on application startup"""
    try:
        # ASCII art and system info
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
        \033[0m
        """)

        # Load the ML model
        predictor.load_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise HTTPException(status_code=500, detail="Model initialization failed")

@app.post("/predict", response_class=JSONResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """Endpoint for plant pathogen detection"""
    try:
        # Validate input
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Process image
        contents = await file.read()
        result = predictor.predict(contents)
        return result

    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_class=JSONResponse)
async def get_model_info():
    """Dynamically extracts model information including all class labels"""
    try:
        if not predictor.model or not predictor.class_labels:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Find base model through layer inspection
        base_model = None
        for layer in predictor.model.layers:
            if isinstance(layer, tf.keras.Model):
                if 'efficientnetv2-b2' in layer.name.lower():
                    base_model = layer
                    break

        if not base_model:
            raise HTTPException(status_code=500,
                              detail="Could not identify base model architecture")

        # Load class names from the JSON file
        with open(f"{predictor.model_path}_class_names.json", 'r') as f:
            class_names = json.load(f)

        # Build response with class information
        info = {
            "input_specification": {
                "shape": predictor.model.input_shape[1:],
                "dtype": "float32",
                "normalization": {
                    "formula": "(x / 127.5) - 1.0",
                    "expected_range": "[-1.0, 1.0]"
                }
            },
            "class_information": {
                "total_classes": len(class_names),
                "all_classes": class_names,
                "class_format": "plant_disease_variant"
            },
            "base_model": {
                "name": base_model.name,
                "trainable_layers": sum(l.trainable for l in base_model.layers),
                "total_parameters": base_model.count_params()
            },
            "prediction_head": {
                "layers": [
                    {
                        "name": layer.name,
                        "type": layer.__class__.__name__,
                        "parameters": layer.count_params()
                    }
                    for layer in predictor.model.layers
                    if layer != base_model
                ]
            },
            "pricing_model": {
                 "base_processing_cost": predictor.BASE_COST,
                 "letterboxing_cost": predictor.LETTERBOX_COST,
                 "cost_per_tta_step": predictor.COST_PER_TTA_STEP,
                 "example_calculation": (
                   f"BASE ({predictor.BASE_COST}) + "
                   f"LETTERBOX ({predictor.LETTERBOX_COST} if needed) + "
                   f"TTA ({predictor.TTA_STEPS} steps × {predictor.COST_PER_TTA_STEP})"
                   )
            }
        }

        return JSONResponse(content=info)

    except Exception as e:
        logger.error(f"Model info error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))