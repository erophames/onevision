import tensorflow as tf
from tensorflow.keras import layers

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
