from typing import Optional

import numpy as np


def lasso(
    predictions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    l1: float,
    eps: Optional[float] = 1e-12,
) -> float:
    """
    Lasso regression objective: MSE + l1 * ||weights||_1

    NOTE: Regularization applies to model weights (NOT predictions).
          Pass weights without the bias term if you do not wish to regularize bias.

    Parameters:
        predictions (np.ndarray): Predicted values.
        targets (np.ndarray): True values (same shape as predictions).
        weights (np.ndarray): Model weights to regularize (any shape).
        l1 (float): L1 regularization factor (>= 0).
        eps (float): Small value to avoid division by zero.

    Returns:
        float: Lasso objective value.
    """
    if not isinstance(predictions, np.ndarray) or not isinstance(targets, np.ndarray):
        raise TypeError("predictions and targets must be numpy arrays.")
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape.")
    if not isinstance(weights, np.ndarray):
        raise TypeError("weights must be a numpy array.")
    if l1 < 0:
        raise ValueError("l1 must be non-negative.")
    if eps is None or not (0 < eps < 1):
        eps = 1e-8

    eps = float(eps)
    predictions = np.clip(predictions, eps, 1.0 - eps)

    mse_part = float(np.mean((predictions - targets) ** 2))
    l1_part = float(l1 * np.sum(np.abs(weights)))
    return mse_part + l1_part
