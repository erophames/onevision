import tensorflow as tf
from tensorflow.keras.metrics import Metric

@tf.keras.utils.register_keras_serializable(package='Custom')
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
    