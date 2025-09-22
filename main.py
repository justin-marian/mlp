import argparse
import sys

import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.mlp import MLPClassifier, MLPRegressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate MLP models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add = p.add_argument
    add("-t","--task", type=str.lower, choices=("classification","regression"), required=True, help="Task type.")
    add("-d","--dataset", type=str.lower, choices=("iris","diabetes"), required=True, help="Dataset.")
    add("-l","--layers", type=int, nargs="+", required=True, help="Hidden layer sizes, e.g. 64 32.")
    add("-a","--activation", type=str.lower, choices=("relu","sigmoid","tanh"), default="relu.")
    add("-decay","--decay_rate", type=float, default=0.01, help="Decay rate for learning rate.")
    add("-lr","--learning_rate", type=float, default=0.01, help="Learning rate.")
    add("--l1", type=float, default=0.0, help="L1 regularization strength (Lasso).")
    add("--l2", type=float, default=0.0, help="L2 regularization strength (Ridge).")
    add("-o","--output", type=str.lower, choices=("softmax","linear"), default="softmax", help="Output head.")
    add("-wi","--weight_init",type=str.lower, choices=("he","xavier","random"), default="he", help="Weight init.")
    add("-e","--epochs", type=int, default=100, help="Training epochs.")
    add("-ts","--test_size", type=float, default=0.2, help="Test split fraction.")
    add("-s","--seed", type=int, default=42, help="Random seed.")
    add("-en","--extension_name", type=str, default="", help="Optional suffix for output files.")
    add("-wf","--wanted_features", type=int, nargs="+", default=None, help="List of feature indices to evaluate.")
    add("--show_loss", action="store_true", help="Print training loss.")
    args = p.parse_args()
    args.output = "linear" if args.task == "regression" else "softmax"
    return args


def classify(params: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, MLPClassifier]:
    """
    Performance metrics on the iris dataset (standardized):
    Accuracy: ~ 95 - 100%
    Precision: ~ 95 - 100%
    Recall: ~ 95 - 100%
    F1-score: ~ 95 - 100%
    """
    iris = load_iris(return_X_y=False)
    X, y = iris.data, iris.target  # type: ignore
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).astype(np.float32).ravel()

    # Guard: base only supports "he" or "xavier"
    wi = params.weight_init.lower()
    if wi == "random":
        wi = "xavier"  # fallback to a safe default

    model = MLPClassifier(
        in_dim=X.shape[1],
        layers=params.layers,
        activation=params.activation,
        learning_rate=params.learning_rate,
        l1=params.l1,
        l2=params.l2,
        decay_rate=params.decay_rate,
        output=params.output,  # always "softmax"
        weight_init=wi,
        seed=params.seed,
        show_loss=params.show_loss,
        extension_name=params.extension_name,
        wanted_features=params.wanted_features,
    )
    return X, y, model


def regress(params: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, MLPRegressor]:
    """
    Performance metrics on the diabetes dataset (standardized):
    R²: ~ 0.40 - 0.55 (linear regression often ~0.45-0.50)
    Explained variance: close to R² (within ~±0.05), also 0.40 - 0.55
    MSE: ~ 3000 - 3500
    RMSE: ~ 55 - 60
    MAE: ~ 40 - 50
    Pearson r: ~ 0.65 - 0.75
    """
    diabetes = load_diabetes(return_X_y=False)
    X, y = diabetes.data, diabetes.target  # type: ignore
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).astype(np.float32).ravel()

    # Guard: base only supports "he" or "xavier"
    wi = params.weight_init.lower()
    if wi == "random":
        wi = "xavier"  # fallback to a safe default

    model = MLPRegressor(
        in_dim=X.shape[1],
        layers=params.layers,
        activation=params.activation,
        learning_rate=params.learning_rate,
        l1=params.l1,
        l2=params.l2,
        decay_rate=params.decay_rate,
        output=params.output,   # always "linear"
        weight_init=wi,
        seed=params.seed,
        show_loss=params.show_loss,
        extension_name=params.extension_name,
        wanted_features=params.wanted_features,
    )
    return X, y, model


if __name__ == "__main__":
    params = parse_args()
    task_to_function = {
        "classification": classify,
        "regression": regress,
    }

    if params.task not in task_to_function:
        print("Error: Invalid task type. Choose 'classification' or 'regression'.")
        sys.exit(1)

    if (params.task == "classification" and params.dataset != "iris") or \
       (params.task == "regression" and params.dataset != "diabetes"):
        print(
            f"Error: The '{params.dataset}' dataset is not supported \n"
            f"for the '{params.task}' task. Please choose the correct dataset."
        )
        sys.exit(1)

    X, y, model = task_to_function[params.task](params)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params.test_size,
        random_state=params.seed
    )
    model.fit(
        X_train, y_train,
        epochs=params.epochs
    )
    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")
    results = model.evaluate(X_test, y_test)
    print("Test Loss:", results['loss'])
    print("Test Metrics:", results['metrics'])
