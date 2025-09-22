import numpy as np


def one_hot_encode(
    labels: list[int] | np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Convert a list of labels to one-hot encoded format.
    The encoding is done on the last dimension.

    Parameters:
        labels (list or np.array): List or array of integer labels.
        num_classes (int): Total number of classes.

    Returns:
        np.array: One-hot encoded representation of the input labels.

    Raises:
        TypeError: If labels is not a list or numpy array of integers.
        ValueError: If any label is out of range [0, num_classes-1].
        ValueError: If num_classes is not a positive integer.
    """
    if not isinstance(labels, (list, np.ndarray)):
        raise TypeError("labels must be a list or numpy array of integers.")
    if not all(isinstance(label, (int, np.integer)) for label in labels):
        raise ValueError("All elements in labels must be integers.")
    if num_classes is not None and (not isinstance(num_classes, int) or num_classes <= 0):
        raise ValueError("num_classes must be a positive integer.")

    pred = np.array(labels).astype(int).ravel()
    C = num_classes if num_classes is not None else int(np.max(pred) + 1)
    if np.any(pred < 0) or np.any(pred >= C):
        raise ValueError("Labels must be in the range [0, num_classes-1].")

    out = np.zeros((pred.size, C), dtype=np.float32)
    out[np.arange(pred.size), pred] = 1.0
    return out
