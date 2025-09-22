from typing import Optional

import numpy as np


def cross_entropy(
    predictions: np.ndarray,
    targets: np.ndarray,
    eps: Optional[float] = 1e-12
) -> float:
    """
    Cross-entropy loss between predictions and targets.

    Parameters:
        predictions (np.ndarray): Predicted probabilities (after softmax),
        shape (N, C) where N is number of samples and C is number of classes.
        targets (np.ndarray): One-hot encoded true labels, shape (N, C).
        eps (float): Small value to avoid log(0).

    Returns:
        float: Cross-entropy loss.
    """
    if not isinstance(predictions, np.ndarray) or not isinstance(targets, np.ndarray):
        raise TypeError("predictions and targets must be numpy arrays.")
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape.")
    if eps is None or not (0 < eps < 1):
        eps = 1e-8

    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, eps, 1. - eps)
    negative_log_likelihood = -np.log(predictions)
    return np.sum(targets * negative_log_likelihood) / predictions.shape[0]
