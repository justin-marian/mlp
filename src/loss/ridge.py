from typing import Optional

import numpy as np


def ridge(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    l2: float,
    eps: Optional[float] = 1e-12,
) -> float:
    """
    Ridge regression objective: MSE + (l2 / 2) * ||weights||_2^2

    NOTE: Regularization applies to model weights (NOT predictions).
          Pass weights without the bias term if you do not wish to regularize bias.

    Parameters:
        predictions (np.ndarray): Predicted values.
        targets (np.ndarray): True values (same shape as predictions).
        weights (np.ndarray): Model weights to regularize (any shape).
        l2 (float): L2 regularization factor (>= 0).
        eps (float): Small value to avoid division by zero.

    Returns:
        float: Ridge objective value.
    """
    if not isinstance(predictions, np.ndarray) or not isinstance(targets, np.ndarray):
        raise TypeError("predictions and targets must be numpy arrays.")
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape.")
    if not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a numpy array.")
    if l2 < 0:
        raise ValueError("l2 must be non-negative.")
    if eps is None or not (0 < eps < 1):
        eps = 1e-8

    eps = float(eps)
    predictions = np.clip(predictions, eps, 1.0 - eps)

    mse_part = float(np.mean((predictions - targets) ** 2))
    l2_part = float((l2 / 2.0) * np.sum(weights ** 2))
    return mse_part + l2_part
