from typing import Literal, Union

import numpy as np

Multioutput = Literal["uniform_average", "variance_weighted", "raw_values"]


def r2_score(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    multioutput: Multioutput = "uniform_average",
    eps: float = 1e-12,
) -> Union[float, np.ndarray]:
    """
    Coefficient of Determination (R^2).

    Parameters:
        predictions (np.ndarray): Predicted values, shape (N,) or (N,1) or (N,D).
        targets (np.ndarray): True values, same shape as predictions.
        multioutput (str): How to aggregate multiple outputs:
            'uniform_average' (default): unweighted mean across outputs.
            'variance_weighted': weighted by variance of each output.
            'raw_values': return R^2 for each output.
        eps (float): Small value to avoid division by zero.

    Returns:
        float: R^2 score (1.0 is perfect prediction).
    """
    if not isinstance(predictions, np.ndarray) or not isinstance(targets, np.ndarray):
        raise TypeError("predictions and targets must be numpy arrays.")
    if eps is None or not (0 < eps < 1):
        eps = 1e-8
    eps = float(eps)

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if not (np.isfinite(predictions).all() and np.isfinite(targets).all()):
        raise ValueError("predictions/targets contain NaN or Inf.")

    # Align to 2D: (N,) -> (N,1)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)

    if predictions.shape[0] != targets.shape[0]:
        raise ValueError(
            f"N mismatch: predictions N={predictions.shape[0]} "
            f"vs targets N={targets.shape[0]}"
        )
    if predictions.shape[1] != targets.shape[1]:
        raise ValueError(
            f"D mismatch: predictions D={predictions.shape[1]} "
            f"vs targets D={targets.shape[1]}"
        )

    # Per-output sums
    y_mean = np.mean(targets, axis=0, keepdims=True)         # (1, D)
    ss_tot = np.sum((targets - y_mean) ** 2, axis=0)         # (D,)
    ss_res = np.sum((targets - predictions) ** 2, axis=0)    # (D,)

    # Handle constant targets per output:
    # If ss_tot == 0:
    #   - if ss_res == 0 -> perfect prediction -> R2 = 1
    #   - else -> R2 = 0
    const_mask = ss_tot <= eps
    r2 = np.empty_like(ss_tot, dtype=float)
    r2[~const_mask] = 1.0 - (ss_res[~const_mask] / (ss_tot[~const_mask] + eps))
    r2[const_mask] = np.where(ss_res[const_mask] <= eps, 1.0, 0.0)

    if multioutput not in ["uniform_average", "variance_weighted", "raw_values"]:
        raise ValueError("multioutput: 'uniform_average', 'variance_weighted', or 'raw_values'")
    elif multioutput == "raw_values":
        return r2
    elif multioutput == "uniform_average":
        return float(np.mean(r2))
    elif multioutput == "variance_weighted":
        weights = np.where(ss_tot > eps, ss_tot, 0.0)
        if np.all(weights == 0):
            return float(np.mean(r2))
        return float(np.average(r2, weights=weights))
