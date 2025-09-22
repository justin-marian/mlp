# -*- coding : utf-8 -*-

"""
Multi-Layer Perceptron (MLP) Base Class.

Caching is used for forward and backward passes.

The forward pass caches:
- Activations
- Pre-activations
- Weights for backpropagation
- Output of each layer for loss computation

The backward pass computes gradients for:
- Pre-activations
- Activations
- Weights
- Biases
- Inputs

Legend:
- A: Activation of the current layer
- Z: Pre-activation (linear output) of the current layer
- W: Weights matrix
- b: Bias vector
- AL: Activation of the Last layer (output)

- dA_prev: Gradient of the loss w.r.t. A from the previous layer
- dA: Gradient of the loss w.r.t. A
- dW: Gradient of the loss w.r.t. W
- db: Gradient of the loss w.r.t. b
- dZ: Gradient of the loss w.r.t. Z
"""

from typing import Optional, Union

import numpy as np

from ..funcs.activations import ActivationFunction
from ..funcs.softmax import softmax
from ..loss.cross_entropy import cross_entropy
from ..loss.lasso import lasso
from ..loss.mae import mae
from ..loss.mse import mse
from ..loss.ridge import ridge
from ..loss.rmse import rmse
from ..utils.classify_metrics import classification_metrics
from ..utils.init_weights import he_init, xavier_init
from ..utils.one_hot import one_hot_encode
from ..utils.plots import plot_function_evaluation, plot_heatmap
from ..utils.r2score import r2_score
from ..utils.variance import explained_variance, pearson_correlation

# Caching type for forward and backward passes in MLP
CACHE_TYPE = list[
    tuple[
        np.ndarray,                      # Activation
        Optional[np.ndarray],            # Pre-Activation
        Optional[np.ndarray],            # Cache for Backward Pass (Weight matrix)
    ]
]


class MLPBase:
    """
    Multi-Layer Perceptron (MLP) for classification and regression tasks.

    - Parameters:
        :in_dim (int): Input feature dimension.
        :layers (list[int]): list with number of neurons in each hidden layer.
        :lr (float): Learning rate.
        :l1 (float): L1 regularization factor.
        :l2 (float): L2 regularization factor.
        :decay_rate (float): Learning rate decay factor per epoch.
        :activation (str): Activation function for hidden layers ('relu', 'sigmoid', 'tanh').
        :output (str): Output layer type ('softmax', 'sigmoid', 'linear').
        :weight_init (str): Weight initialization method ('xavier', 'he').
        :seed (Optional[int]): Random seed for reproducibility.
        :verbose (bool): Whether to print model configuration on initialization.
        :extension_name (str): Optional suffix for output files.
        :wanted_features (Optional[list[int]]): list of feature indices to evaluate after training.

    - Methods:
        :fit: Train the MLP on input data and labels.
        :predict: Make predictions on new input data.
        :evaluate: Evaluate the model on test data and labels.
        :forward: Perform a forward pass through the network.
        :backward: Perform a backward pass and compute gradients.
        :l2_penalty: Compute the L2 regularization penalty.
        :lr_decay: Apply learning rate decay.

    - Example:
    ```python
        mlp = MLP(
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
            extension_name='experiment1'
            wanted_features=[0, 1]
        )
    ```
    """

    TYPES_OUTPUT: list[str] = [
        "softmax",
        "sigmoid",
        "linear"
    ]

    def __init__(
        self,
        in_dim: int,
        layers: list[int],
        lr: float = 1e-3,
        l1: float = 0.0,
        l2: float = 0.0,
        decay_rate: float = 0.001,
        activation: str = "relu",
        output: str = "softmax",
        weight_init: str = "xavier",
        seed: Optional[int] = None,
        verbose: bool = True,
        extension_name: str = "",
        wanted_features: Optional[list[int]] = None,
    ) -> None:
        self.in_dim = in_dim
        self.layers = layers
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self.activation = activation
        self.output = output
        self.weight_init = weight_init
        self.decay_rate = decay_rate
        self.extension_name = extension_name
        self.wanted_features = wanted_features

        self.seed = seed
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        if weight_init not in ["xavier", "he"]:
            raise ValueError("weight_init must be 'xavier' or 'he'")
        self.weight_init_func = xavier_init if weight_init == "xavier" else he_init

        if activation not in ["relu", "sigmoid", "tanh"]:
            raise ValueError("activation must be 'relu', 'sigmoid' or 'tanh'")
        self.activ_func = {
            "relu": ActivationFunction.relu,
            "sigmoid": ActivationFunction.sigmoid,
            "tanh": ActivationFunction.tanh,
        }[activation]
        self.activ_deriv = {
            "relu": ActivationFunction.d_relu,
            "sigmoid": ActivationFunction.d_sigmoid,
            "tanh": ActivationFunction.d_tanh,
        }[activation]

        if output not in self.TYPES_OUTPUT:
            raise ValueError(f"output must be one of {self.TYPES_OUTPUT}")

        if verbose:
            print(
                f"Initialized MLP with layers: {self.layers}, "
                f"activation: {self.activation}, "
                f"output: {self.output}, "
                f"weight_init: {self.weight_init}",
                end="\n", flush=True, sep=""
            )

        # Weight initialization
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.caches: CACHE_TYPE = []
        self.input_batch: np.ndarray = np.empty((0,))
        self.pred_batch: np.ndarray = np.empty((0,))

        dims = [self.in_dim] + self.layers
        for i in range(len(dims) - 1):
            fa_in, fa_out = dims[i], dims[i + 1]
            self.W.append(self.weight_init_func(fa_in, fa_out, rng=self.rng))
            self.b.append(np.zeros((1, fa_out), dtype=np.float32))

        self.no_classes = self.layers[-1] if output in ["softmax", "sigmoid"] else 1

        if verbose:
            print(f"Weight shapes: {[w.shape for w in self.W]}")
            print(f"Bias shapes: {[b.shape for b in self.b]}")
            print(f"Number of output classes: {self.no_classes}")


    def predict(self, inputs: np.ndarray) -> np.ndarray:
        prob_mid = 0.5  # threshold for binary classification
        AL = self.forward(inputs)
        if self.output == "softmax":
            return np.argmax(AL, axis=1)                        # (N,)
        elif self.output == "sigmoid":
            return (AL >= prob_mid).astype(np.int32).reshape(-1, 1)  # (N,1)
        else:
            return AL                                           # (N, out_dim)


    def l2_penalty(self) -> float:
        return (self.l2 / 2) * sum(np.sum(W ** 2) for W in self.W)


    def lr_decay(self, decay_rate: float) -> None:
        self.lr *= (1.0 / (1.0 + decay_rate))


    def exploding_grads(self, *arrays) -> bool:
        return any((not np.isfinite(a).all()) for a in arrays if a is not None)


    def clip_grads(
        self,
        grads_W: list[np.ndarray],
        grads_b: list[np.ndarray],
        clip_value: Optional[float] = 1.0
    ) -> None:
        if clip_value is None or clip_value <= 0:
            return
        for g in grads_W:
            np.clip(g, -clip_value, clip_value, out=g)
        for g in grads_b:
            np.clip(g, -clip_value, clip_value, out=g)


    def forward(
        self,
        inputs: np.ndarray
    ) -> np.ndarray:
        # Caching activations and pre-activations for backprop
        A = inputs.astype(np.float32)
        caches: CACHE_TYPE = [(A, None, None)]
        L = len(self.W)

        # Hidden layers
        for layer in range(L - 1):
            Z = A @ self.W[layer] + self.b[layer]  # (N, H)
            A = self.activ_func(Z)
            caches.append((A, Z, self.W[layer]))

        # Final layer
        ZL = A @ self.W[-1] + self.b[-1]  # (N, C)
        AL = {
            "softmax": softmax(ZL),
            "sigmoid": ActivationFunction.sigmoid(ZL),
            "linear": ZL,
        }[self.output]

        caches.append((AL, ZL, self.W[-1]))
        self.caches = caches
        return AL


    def backward(
        self,
        caches: CACHE_TYPE,
        AL: np.ndarray,
        preds: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        grads_W = [np.zeros_like(W) for W in self.W]
        grads_b = [np.zeros_like(b) for b in self.b]

        N = preds.shape[0]
        # Compute dZ for last layer depending on output/loss
        preds = preds.astype(np.float32)
        AL = AL.astype(np.float32)

        # dZ depends on output layer and loss function used
        dZ: np.ndarray = {
            "softmax": (AL - preds) / N,
            "sigmoid": (AL - preds) / N,
            "linear":  (2.0 / N) * (AL - preds),
            "mse":     (2.0 / N) * (AL - preds),
            "mae":     np.sign(AL - preds) / N,
            "rmse":    (AL - preds) / (N * (np.sqrt(np.mean((AL - preds) ** 2)) + 1e-8)),
            "lasso":   (2.0 / N) * (AL - preds),
            "ridge":   (2.0 / N) * (AL - preds),
        }[self.output]

        # Last layer
        A_prev = caches[-2][0]
        grads_W[-1] = (A_prev.T @ dZ)  # (H, C)
        grads_b[-1] = np.sum(dZ, axis=0, keepdims=True)

        # Hidden layers
        for layer in range(len(self.W) - 2, -1, -1):
            dA_prev = dZ @ self.W[layer + 1].T
            Z_l = caches[layer + 1][1]            # (N, H)
            A_prev = caches[layer][0]             # (N, H)

            if Z_l is None:
                raise ValueError("Pre-activation Z is None in cache -> cannot compute gradients.")

            dZ = dA_prev * self.activ_deriv(Z_l)  # (N, H)
            grads_W[layer] = (A_prev.T @ dZ)      # (H_prev, H)
            grads_b[layer] = np.sum(dZ, axis=0, keepdims=True)

        # Regularization terms
        if self.l2 > 0 and self.l1 <= 0:  # Ridge
            for layer in range(len(self.W)):
                grads_W[layer] += self.l2 * self.W[layer]
        elif self.l1 > 0:                 # Lasso (subgradient)
            for layer in range(len(self.W)):
                grads_W[layer] += self.l1 * np.sign(self.W[layer])

        return grads_W, grads_b, dZ


    def fit(
        self,
        inputs: np.ndarray,
        preds: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
        shuffle: bool = True,
        *,
        log_every: int = 10,
        patience: int = 0,
        min_delta: float = 0.0,
        grad_clip_value: float = 1.0,
    ) -> None:
        self.epochs = epochs
        inputs = np.asarray(inputs, dtype=np.float32)
        preds = np.asarray(preds,  dtype=np.float32)

        num_samples = inputs.shape[0]
        batch_size = min(batch_size, num_samples)
        output_dim = self.W[-1].shape[1]

        # Normalize targets to the head
        preds = {
            "softmax": one_hot_encode(preds.astype(int), num_classes=output_dim).astype(np.float32),
            "sigmoid": one_hot_encode(preds.astype(int), num_classes=output_dim).astype(np.float32),
            "linear": preds.reshape(-1, output_dim).astype(np.float32),
        }[self.output]

        best_loss = np.inf
        patience_left = patience

        for epoch in range(1, epochs + 1):
            idx = np.arange(num_samples)
            if shuffle:
                self.rng.shuffle(idx)

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = idx[start:end]
                input_batch = inputs[batch_idx]
                target_batch = preds[batch_idx]

                AL = self.forward(input_batch)

                if self.exploding_grads(AL):
                    if verbose:
                        print("Exploding/NaN gradients detected, reducing learning rate")
                    self.lr *= 0.1
                    AL = np.nan_to_num(AL, nan=0.0, posinf=1e6, neginf=-1e6)

                loss: float = {
                    # Depending on output layer type
                    "softmax": cross_entropy(AL, target_batch) + self.l2_penalty(),  # CCE
                    "sigmoid": cross_entropy(AL, target_batch) + self.l2_penalty(),  # BCE
                    "linear":  mse(AL, target_batch) + self.l2_penalty(),  # MSE
                    # Special case for regression losses
                    "mse":    mse(AL, target_batch) + self.l2_penalty(),   # MSE
                    "mae":    mae(AL, target_batch) + self.l2_penalty(),   # MAE
                    "rmse":   rmse(AL, target_batch) + self.l2_penalty(),  # RMSE
                    # Regularization losses only
                    "lasso":  lasso(AL, target_batch, self.W[0], self.l1),  # L1
                    "ridge":  ridge(AL, target_batch, self.W[0], self.l2),  # L2
                }[self.output]

                grads_W, grads_b, _ = self.backward(self.caches, AL, target_batch)
                # Clip grads to avoid explosions
                self.clip_grads(grads_W, grads_b, clip_value=grad_clip_value)

                # Weight update
                for layer in range(len(self.W)):
                    self.W[layer] -= self.lr * grads_W[layer]
                    self.b[layer] -= self.lr * grads_b[layer]

                epoch_loss += float(loss)
                num_batches += 1

            epoch_loss /= max(1, num_batches)

            # Logging
            self.show_loss = (epoch % log_every == 0) or (epoch == epochs)
            if verbose and self.show_loss:
                print(f"Epoch {epoch}/{self.epochs}  |  loss={epoch_loss:.6f}")

            # Early stopping
            if patience > 0:
                if best_loss - epoch_loss > min_delta:
                    best_loss = epoch_loss
                    patience_left = patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if verbose:
                            print(f"Early stopping at epoch {epoch} (best loss {best_loss:.6f})")
                        break

            self.lr_decay(self.decay_rate)

        # Post-training plots (skip if non-finite)
        if self.in_dim > 1:
            mu = inputs.mean(axis=0, keepdims=True).astype(np.float32)
            if self.wanted_features is None:
                features_to_plot = list(range(min(self.in_dim, 2)))
            else:
                features_to_plot = [i for i in self.wanted_features if i < self.in_dim]

            for idx in features_to_plot:
                # Generate a grid of values for the selected feature
                xs = np.linspace(inputs[:, idx].min(), inputs[:, idx].max(), 200, dtype=np.float32)
                grid = np.repeat(mu, len(xs), axis=0)
                grid[:, idx] = xs
                yhat = self.forward(grid)

                # If predictions are not finite, skip the plot
                if self.exploding_grads(yhat):
                    if verbose:
                        print(f"Exploding gradients for feature {idx}, skipping plot")
                    continue

                ext_suffix = f"_{self.extension_name}" if self.extension_name else ""
                save_path = f"outputs/eval_curve_feature{idx}{ext_suffix}.png"

                plot_function_evaluation(
                    x=xs,
                    y=yhat,
                    title=f"Evaluation MLP (feature {idx}){ext_suffix}",
                    xlabel=f"Feature {idx}",
                    ylabel="Predicted Output",
                    save_path=save_path,
                    mode="probs" if self.output == "softmax" else "auto",
                    class_names=(
                        ["setosa", "versicolor", "virginica"]
                        if output_dim == self.no_classes else None
                    ),
                    show_threshold=True
                )


    def evaluate(
        self,
        inputs: np.ndarray,
        preds: np.ndarray,
    ) -> dict:
        dim_deriv = 2  # for 1D arrays
        inputs = np.asarray(inputs).astype(np.float32)
        trues = np.asarray(preds).copy()
        preds = np.asarray(preds).astype(np.float32)

        # Normalize targets to the head format
        if self.output == "softmax":
            # Multi-class classification head
            if preds.ndim == 1 or (preds.ndim == dim_deriv and preds.shape[1] == 1):
                preds_oh = one_hot_encode(
                    preds.astype(int), num_classes=self.layers[-1]
                ).astype(np.float32)
            else:
                preds_oh = preds.astype(np.float32)
        elif self.output == "sigmoid":
            # Binary classification head
            preds_oh = preds.reshape(-1, 1).astype(np.float32)
        else:  # "linear"
            # One output regression head
            preds_oh = preds.reshape(-1, self.layers[-1]).astype(np.float32)

        AL = self.forward(inputs)

        loss: float = {
            # Depending on output layer type
            "softmax": cross_entropy(AL, preds_oh) + self.l2_penalty(),  # CCE
            "sigmoid": cross_entropy(AL, preds_oh) + self.l2_penalty(),  # BCE
            "linear":  mse(AL, preds_oh) + self.l2_penalty(),  # MSE
            # Special case for regression metrics
            "mse":    mse(AL, preds_oh) + self.l2_penalty(),   # MSE
            "mae":    mae(AL, preds_oh) + self.l2_penalty(),   # MAE
            "rmse":   rmse(AL, preds_oh) + self.l2_penalty(),  # RMSE
            # Regularization losses only
            "lasso":  lasso(AL, preds_oh, self.W[0], self.l1),  # L1
            "ridge":  ridge(AL, preds_oh, self.W[0], self.l2),  # L2
        }[self.output]

        # For storing metrics - classification or regression
        metrics: Union[dict[str, float], tuple[float, float, float, float]] = {}

        # Metrics per task
        if self.output in ("softmax", "sigmoid"):
            y_pred_labels = self.predict(inputs)  # labels (N,) or (N,1)
            metrics = classification_metrics(trues, y_pred_labels, self.output)
        else:
            metrics = {
                "mse": float(mse(AL, preds_oh)),
                "mae": float(mae(AL, preds_oh)),
                "rmse": float(rmse(AL, preds_oh)),
            }
            r2 = r2_score(AL, trues)
            ev = explained_variance(AL, trues)
            corr = pearson_correlation(AL, trues)

            print(f"R²: {r2:.4f}")
            print(f"Explained Variance: {ev:.4f}")
            print(f"Pearson Correlation: {corr:.4f}")

        if getattr(self, "show_loss", False):
            print(f"Loss: {loss:.4f}, Metrics: {metrics}")

        # Only for classification tasks
        if self.output == "softmax" and AL.ndim == dim_deriv and AL.shape[1] > 1:
            plot_heatmap(
                data=AL,
                title="Output Layer Activations (Softmax) Heatmap",
                xlabel="Classes",
                ylabel="Samples"
            )
        return {"loss": float(loss), "metrics": metrics}
