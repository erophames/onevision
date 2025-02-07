import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils import class_weight
from .augmentation import create_augmentation_pipeline

def create_data_pipeline(dataset_path, img_size, batch_size, validation_split, seed):
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
    data_augmentation = create_augmentation_pipeline()

    # Load datasets
    original_train_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='int'
    )

    original_val_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='int'
    )

    # Save class labels from original dataset
    class_labels = original_train_ds.class_names

    # Calculate class weights
    train_labels = np.concatenate([y for x, y in original_train_ds], axis=0)
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights))

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

    return train_ds, val_ds, class_labels, class_weights
