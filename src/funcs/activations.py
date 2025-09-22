import numpy as np


class ActivationFunction:
    ###############################################################################
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def d_relu(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(x.dtype)
    ###############################################################################
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        pos = x >= 0
        neg = ~pos
        out = np.empty_like(x)
        out[pos] = 1 / (1 + np.exp(-x[pos]))
        expz = np.exp(x[neg])
        out[neg] = expz / (1 + expz)
        return out

    @staticmethod
    def d_sigmoid(x: np.ndarray) -> np.ndarray:
        sig = ActivationFunction.sigmoid(x)
        return sig * (1 - sig)
    ###############################################################################
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def d_tanh(x: np.ndarray) -> np.ndarray:
        t = np.tanh(x)
        return 1 - t * t
    ###############################################################################
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def d_linear(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    ###############################################################################
