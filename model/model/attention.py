import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

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
