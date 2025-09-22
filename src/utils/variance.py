# -*- coding : utf-8 -*-

import numpy as np


def explained_variance(
    predictions: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-12
) -> float:
    """
    NOTE: Explained Variance (EV) is a statistical measure that quantifies the
    proportion of  variance in the target variable that is predictable from the
    input features. It is commonly used to evaluate the performance of regression
    models. EV values range from 0 to 1, where 1 indicates perfect prediction and 0
    indicates that the model does not explain any of the variance in the target variable.
    """
    targets = np.asarray(targets).reshape(-1, 1) if targets.ndim == 1 else targets
    predictions = np.asarray(predictions).reshape(-1, 1) if predictions.ndim == 1 else predictions

    num = np.var(targets - predictions, axis=0)
    den = np.var(targets, axis=0) + eps
    ev = 1.0 - (num / den)

    return float(np.mean(ev))


def pearson_correlation(
    predictions: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-12
) -> float:
    """
    NOTE: Pearson correlation coefficient (r) is a statistical measure that quantifies
    the strength and direction of the linear relationship between two variables. It
    ranges from -1 to 1, where 1 indicates a perfect positive linear relationship,
    -1 indicates a perfect negative linear relationship, and 0 indicates no linear
    relationship.
    """
    targets = np.asarray(targets).reshape(-1, 1) if targets.ndim == 1 else targets
    predictions = np.asarray(predictions).reshape(-1, 1) if predictions.ndim == 1 else predictions

    mean_pred = np.mean(predictions, axis=0)
    mean_true = np.mean(targets, axis=0)

    cov = np.mean((predictions - mean_pred) * (targets - mean_true), axis=0)
    std_pred = np.std(predictions, axis=0) + eps
    std_true = np.std(targets, axis=0) + eps
    corr = cov / (std_pred * std_true)

    return float(np.mean(corr))
