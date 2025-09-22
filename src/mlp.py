from typing import Optional

import numpy as np

from .base.mlp_base import MLPBase


class MLPRegressor(MLPBase):
    """
    MLPRegressor is a multi-layer perceptron for regression tasks.
    It supports multiple hidden layers, various activation functions, and
    different output layer configurations.

    Parameters:
        :in_dim (int): Number of input features.
        :layers (List[int]): List containing the number of neurons in each hidden layer.
        :activation (str): Activation function for hidden layers ('relu', 'tanh', 'sigmoid').
        :learning_rate (float): Learning rate for weight updates.
        :output (str): Output layer activation function ('linear' or 'sigmoid').
        :weight_init (str): Weight initialization method ('he', 'xavier', 'random').
        :seed (Optional[int]): Random seed for reproducibility.
        :show_loss (bool): Whether to display loss during training.

    Returns: None

    Example:
    ```python
        from .mlp import MLPRegressor
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_diabetes
        from sklearn.preprocessing import StandardScaler

        # Initialize MLPRegressor
        mlp = MLPRegressor(
            in_dim=20,
            layers=[64, 32, 10],
            lr=0.001,
            l1=0.01,
            l2=0.01,
            decay_rate=0.001,
            activation='relu',
            output='softmax',
            weight_init='xavier',
            seed=42,
            verbose=True,
            extension_name='regression_experiment',
            wanted_features=[0, 1]
        )
        # Prepare data (example with diabetes dataset)
        scaler = StandardScaler()
        X, y = load_diabetes(return_X_y=True)
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )
        # Train the model
        mlp.fit(
            X_train, y_train,
            epochs=100, batch_size=16,
            verbose=True, shuffle=True,
            one_hot=False
        )
        print(
            f"Training complete: {mlp.epochs} epochs, "
            f"learning rate: {mlp.lr}, L2: {mlp.l2}, "
            f"activation: {mlp.activation}, output: {mlp.output}, "
            f"predictions: {mlp.predict(X_test)}"
        )
        # Evaluate on test data
        results = mlp.evaluate(X_test, y_test, one_hot=False)
        print("Test Loss:", results['loss'])
    ```
    """
    def __init__(
        self,
        in_dim: int,
        layers: list[int],
        activation: str = "relu",
        learning_rate: float = 0.01,
        l1: float = 0.0,
        l2: float = 0.0,
        decay_rate: float = 0.001,
        output: str = "linear",
        weight_init: str = "he",
        seed: Optional[int] = None,
        show_loss: bool = False,
        extension_name: str = "",
        wanted_features: Optional[list[int]] = None,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            layers=layers,
            lr=learning_rate,
            l1=l1,
            l2=l2,
            decay_rate=decay_rate,
            activation=activation,
            output=output,
            weight_init=weight_init,
            seed=seed,
            verbose=show_loss,
            extension_name=extension_name,
            wanted_features=wanted_features,
        )
        if output not in ["linear", "sigmoid"]:
            raise ValueError("Output layer must be 'linear' or 'sigmoid' for regression tasks.")


    def fit(self, inputs: np.ndarray, preds: np.ndarray, *args, **kwargs) -> None:
        super().fit(inputs=inputs, preds=preds, *args, **kwargs)


    def evaluate(
        self,
        inputs: np.ndarray,
        preds: np.ndarray,
    ) -> dict[str, float]:
        results = super().evaluate(inputs, preds)
        return results


class MLPClassifier(MLPBase):
    """
    MLPClassifier is a multi-layer perceptron for classification tasks.
    It supports multiple hidden layers, various activation functions, and
    different output layer configurations.

    Parameters:
        :in_dim (int): Number of input features.
        :layers (List[int]): List containing the number of neurons in each hidden layer.
        :activation (str): Activation function for hidden layers ('relu', 'tanh', 'sigmoid').
        :learning_rate (float): Learning rate for weight updates.
        :output (str): Output layer activation function ('softmax': multi-class, 'sigmoid': binary).
        :weight_init (str): Weight initialization method ('he', 'xavier', 'random').
        :seed (Optional[int]): Random seed for reproducibility.
        :show_loss (bool): Whether to display loss during training.

    Returns: None

    Example:
    ```python
        from .mlp import MLPClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler

        # Initialize MLP
        mlp = MLPClassifier(
            in_dim=20,
            layers=[64, 32, 10],
            lr=0.001,
            l1=0.01,
            l2=0.01,
            decay_rate=0.001,
            activation='relu',
            output='softmax',
            weight_init='xavier',
            seed=42,
            verbose=True,
            extension_name='classification_experiment',
            wanted_features=[0, 1]
        )
        # Prepare data (example with Iris dataset)
        scaler = StandardScaler()
        X, y = load_iris(return_X_y=True)
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )
        # Train the model
        mlp.fit(
            X_train, y_train,
            epochs=100, batch_size=16,
            verbose=True, shuffle=True,
            one_hot=True
        )
        print(
            f"Training complete: {mlp.epochs} epochs, "
            f"learning rate: {mlp.lr}, L2: {mlp.l2}, "
            f"activation: {mlp.activation}, output: {mlp.output}, "
            f"predictions: {mlp.predict(X_test)}"
        )
        # Evaluate on test data
        results = mlp.evaluate(X_test, y_test, one_hot=True)
        print("Test Loss:", results['loss'])
        print("Test Metrics:", results['metrics'])
    ```
    """

    def __init__(
        self,
        in_dim: int,
        layers: list[int],
        activation: str = "relu",
        learning_rate: float = 0.01,
        l1: float = 0.0,
        l2: float = 0.0,
        decay_rate: float = 0.001,
        output: str = "softmax",
        weight_init: str = "he",
        seed: Optional[int] = None,
        show_loss: bool = False,
        extension_name: str = "",
        wanted_features: Optional[list[int]] = None,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            layers=layers,
            lr=learning_rate,
            l1=l1,
            l2=l2,
            decay_rate=decay_rate,
            activation=activation,
            output=output,
            weight_init=weight_init,
            seed=seed,
            verbose=show_loss,
            extension_name=extension_name,
            wanted_features=wanted_features,
        )
        if output not in ["softmax", "sigmoid"]:
            raise ValueError("Output layer must be 'softmax'/'sigmoid' for classification tasks.")


    def fit(self, inputs: np.ndarray, preds: np.ndarray, *args, **kwargs) -> None:
        super().fit(inputs=inputs, preds=preds, *args, **kwargs)
