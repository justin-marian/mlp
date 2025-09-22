from typing import Optional

import numpy as np


def mse(
    predictions: np.ndarray,
    targets: np.ndarray,
    eps: Optional[float] = 1e-12
) -> float:
    """
    Mean Squared Error (MSE) between predictions and targets.

    Parameters:
        predictions (np.ndarray): Predicted values.
        targets (np.ndarray): True values.
        eps (float): Small value to avoid division by zero.

    Returns:
        float: Mean Squared Error.
    """
    if not isinstance(predictions, np.ndarray) or not isinstance(targets, np.ndarray):
        raise TypeError("predictions and targets must be numpy arrays.")
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape.")
    if eps is None or not (0 < eps < 1):
        eps = 1e-8

    # Clip predictions to avoid division by zero
    eps = float(eps) if eps is not None else 1e-12
    predictions = np.clip(predictions, eps, 1.0 - eps)
    return float(np.mean((predictions - targets) ** 2))
