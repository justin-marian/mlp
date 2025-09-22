import numpy as np


def flatten(W: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Flatten a list of weight matrices into a single vector.
    Also returns the original shapes for unflattening.

    Parameters:
        W (list of np.ndarray): List of weight matrices to flatten.

    Returns:
        tuple: (flattened_weights, original_shapes)
            flattened_weights (np.ndarray): 1D array of all weights concatenated.
            original_shapes (list of tuple): List of (rows, cols) for each weight matrix.
    """
    if not isinstance(W, list) or not all(isinstance(w, np.ndarray) for w in W):
        raise TypeError("W must be a list of numpy arrays.")

    original_shapes = [(w.shape[0], w.shape[1]) for w in W]
    flattened_weights = np.concatenate([w.ravel() for w in W])

    return flattened_weights, original_shapes
