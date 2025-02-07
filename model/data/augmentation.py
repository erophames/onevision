import tensorflow as tf
from tensorflow.keras import layers

def create_augmentation_pipeline():
    """Creates a data augmentation pipeline using Keras Sequential API.

        This function defines a series of random augmentations to be applied to the input images
        during training. The augmentations help simulate real-world variations and improve the
        model's ability to generalize.

        Returns:
            tf.keras.Sequential: A sequential model containing the data augmentation layers.
        """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ])
    return data_augmentation
