from typing import Optional

import numpy as np


def soft_labels(
    labels: list[int] | np.ndarray,
    num_classes: Optional[int] = None,
    smoothing: float = 0.1
) -> np.ndarray:
    """
    Convert a list of labels to smoothed soft label format.
    The smoothing is done on the last dimension.

    NOTE: This is typically used for classification tasks to improve model generalization.

    Parameters:
        labels (list or np.array): List or array of integer labels.
        num_classes (int, optional): Total number of classes. If None,
        it will be inferred from the labels.
        smoothing (float, optional): Smoothing factor between 0 and 1. Defaults to 0.1.

    Returns:
        np.array: Smoothed soft label representation of the input labels.

    Raises:
        TypeError: If labels is not a list or numpy array of integers.
        ValueError: If any label is out of range [0, num_classes-1].
        ValueError: If num_classes is not a positive integer.
        ValueError: If smoothing is not in the range [0, 1).
    """
    if not isinstance(labels, (list, np.ndarray)):
        raise TypeError("labels must be a list or numpy array of integers.")
    if not all(isinstance(label, (int, np.integer)) for label in labels):
        raise ValueError("All elements in labels must be integers.")
    if num_classes is not None and (not isinstance(num_classes, int) or num_classes <= 0):
        raise ValueError("num_classes must be a positive integer.")
    if not (0 <= smoothing < 1):
        raise ValueError("smoothing must be in the range [0, 1).")

    pred = np.array(labels).astype(int).ravel()
    C = num_classes if num_classes is not None else int(np.max(pred) + 1)
    if np.any(pred < 0) or np.any(pred >= C):
        raise ValueError("Labels must be in the range [0, num_classes-1].")

    out = np.full((pred.size, C), smoothing / (C - 1), dtype=np.float32)
    out[np.arange(pred.size), pred] = 1.0 - smoothing
    return out
