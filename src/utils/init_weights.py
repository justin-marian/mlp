import numpy as np


def xavier_init(
    fa_in: int,
    fa_out: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Xavier initialization of network weights.

    Parameters:
        fa_in (int): Number of input features.
        fa_out (int): Number of output features.
        rng (np.random.Generator): Numpy random number generator instance.

    Returns:
        np.ndarray: Initialized weight matrix of shape (fa_in, fa_out).

    Raises:
        ValueError: If fa_in or fa_out is not a positive integer.
        TypeError: If rng is not an instance of np.random.Generator.

    References:
        - Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of
        training deep feedforward neural networks. In Proceedings of the thirteenth
        international conference on artificial intelligence and statistics (pp. 249-256).
        - https://en.wikipedia.org/wiki/Weight_initialization#Xavier_initialization
    """
    if not isinstance(fa_in, int) or fa_in <= 0:
        raise ValueError("fa_in must be a positive integer.")
    if not isinstance(fa_out, int) or fa_out <= 0:
        raise ValueError("fa_out must be a positive integer.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of np.random.Generator.")

    limit = np.sqrt(6 / (fa_in + fa_out))
    return rng.uniform(-limit, limit, size=(fa_in, fa_out)).astype(np.float32)


def he_init(
    fa_in: int,
    fa_out: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    He initialization of network weights.

    Parameters:
        fa_in (int): Number of input features.
        fa_out (int): Number of output features.
        rng (np.random.Generator): Numpy random number generator instance.

    Returns:
        np.ndarray: Initialized weight matrix of shape (fa_in, fa_out).

    Raises:
        ValueError: If fa_in or fa_out is not a positive integer.
        TypeError: If rng is not an instance of np.random.Generator.

    References:
        - He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into
        rectifiers: Surpassing human-level performance on imagenet classification.
        In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034).
        - https://en.wikipedia.org/wiki/Weight_initialization#He_initialization
    """
    if not isinstance(fa_in, int) or fa_in <= 0:
        raise ValueError("fa_in must be a positive integer.")
    if not isinstance(fa_out, int) or fa_out <= 0:
        raise ValueError("fa_out must be a positive integer.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of np.random.Generator.")

    stddev = np.sqrt(2 / fa_in)
    return rng.normal(0, stddev, size=(fa_in, fa_out)).astype(np.float32)
