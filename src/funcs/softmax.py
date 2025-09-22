from typing import Optional

import numpy as np


def softmax(
    logits: np.ndarray,
    axis: Optional[int] = -1
) -> np.ndarray:
    """
    Softmax of each element along an axis of the input array.

    Parameters:
        logits (np.ndarray): Input array.
        axis (int): Axis along which the softmax is computed. Default is -1 (last axis).

    Returns:
        np.ndarray: Softmax of the input array along the specified axis.

    Raises:
        TypeError: If logits is not a numpy array.
        ValueError: If axis is out of bounds for the input array dimensions.
    """
    if not isinstance(logits, np.ndarray):
        raise TypeError("logits must be a numpy array.")

    pick_axis = axis if axis is not None else -1
    if pick_axis < -logits.ndim or pick_axis >= logits.ndim:
        raise ValueError(f"axis {pick_axis} is out of bounds for array of dimension {logits.ndim}.")

    # Numerical stability by subtracting maximum logit
    shifted_logits = logits - np.max(logits, axis=pick_axis, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    sum_exp = np.sum(exp_logits, axis=pick_axis, keepdims=True)
    return exp_logits / sum_exp
