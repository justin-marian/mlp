import numpy as np
from sklearn.metrics import (
    accuracy_score,  # (TP + TN) / (TP + TN + FP + FN)
    f1_score,  # 2 * (precision * recall) / (precision + recall)
    precision_score,  # TP / (TP + FP)
    recall_score,  # TP / (TP + FN)
)


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    output: str
) -> tuple[float, float, float, float]:
    """
    Basic metrics for classification tasks.

    Accepts either probabilities or labels on both sides and
    converts them to label vectors consistently.

    Args:
        targets: ground-truth; shape (N,), (N,1), or one-hot (N,C)
        predictions: predictions; labels (N,) or probs: (N,C) for softmax, (N,1)/(N,) for sigmoid
        output: "softmax" (multi-class) or "sigmoid" (binary)

    Returns:
        (accuracy, precision_weighted, recall_weighted, f1_weighted)
    """
    dim = 2  # for 2D arrays
    prob_mid = 0.5  # threshold for sigmoid
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    if output == "softmax":
        if targets.ndim == dim and targets.shape[1] > 1:
            targets_lbl = np.argmax(targets, axis=1)
        else:
            targets_lbl = targets.reshape(-1)

        if predictions.ndim == dim and predictions.shape[1] > 1:
            predictions_lbl = np.argmax(predictions, axis=1)
        else:
            predictions_lbl = predictions.reshape(-1)

        average = "weighted"

    elif output == "sigmoid":
        if targets.ndim == dim and targets.shape[1] == 1:
            targets_lbl = targets[:, 0]
        else:
            targets_lbl = targets.reshape(-1)
        targets_lbl = (
            targets_lbl >= prob_mid
        ).astype(int) if targets_lbl.dtype != np.int_ else targets_lbl.astype(int)

        if predictions.ndim == dim and predictions.shape[1] == 1:
            predictions_lbl = (predictions[:, 0] >= prob_mid).astype(int)
        else:
            predictions_lbl = predictions.reshape(-1)
            if predictions_lbl.dtype not in {np.int32, np.int64}:
                predictions_lbl = (predictions_lbl >= prob_mid).astype(int)

        average = "binary"
    else:
        raise ValueError(f"Unsupported output type for classification metrics: {output}")

    acc = accuracy_score(predictions_lbl, targets_lbl)
    prec = precision_score(targets_lbl, predictions_lbl, average=average, zero_division=0)
    rec = recall_score(targets_lbl, predictions_lbl, average=average, zero_division=0)
    f1 = f1_score(targets_lbl, predictions_lbl, average=average, zero_division=0)
    return float(acc), float(prec), float(rec), float(f1)
