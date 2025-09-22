# Multi-Layer Perceptron

Multi-Layer Perceptron (MLP) implemented with NumPy that can tackle both classification and regression tasks. It’s designed to be small, readable, and hackable, while still covering the practical pieces you need to run credible experiments: 

- **solid weight initialization**
- **multiple activations**
- **L1/L2 regularization** 
- **learning-rate decay**
- **gradient clipping**
- **proper target shaping**
- **standardized evaluation metrics** 
- **quick visualization utilities**

The MLP follows a standard feedforward architecture with:

- an input layer of dimension n
- multiple hidden layers with configurable sizes
- and an output layer of dimension k suitable for the specific task

## Overview

MLP implementation provides a flexible framework for neural network training with:
- **Classification**: Multi-class classification using Iris dataset
- **Regression**: Continuous value prediction using Diabetes dataset
- **Regularization**: L1 (Lasso) and L2 (Ridge) regularization
- **Multiple Activations**: ReLU, Sigmoid, Tanh
- **Advanced Features**: Learning rate decay, gradient clipping, early stopping

## Console Arguments

| Argument | Short | Type | Choices | Default | Description |
|----------|-------|------|---------|---------|-------------|
| `--task` | `-t` | str | classification, regression | Required | Task type |
| `--dataset` | `-d` | str | iris, diabetes | Required | Dataset to use |
| `--layers` | `-l` | int+ | - | Required | Hidden layer sizes |
| `--activation` | `-a` | str | relu, sigmoid, tanh | relu | Activation function |
| `--decay_rate` | `-decay` | float | - | 0.01 | Learning rate decay |
| `--learning_rate` | `-lr` | float | - | 0.01 | Initial learning rate |
| `--l1` | - | float | - | 0.0 | L1 regularization |
| `--l2` | - | float | - | 0.0 | L2 regularization |
| `--output` | `-o` | str | softmax, linear | auto | Output layer type |
| `--weight_init` | `-wi` | str | he, xavier, random | he | Weight initialization |
| `--epochs` | `-e` | int | - | 100 | Training epochs |
| `--test_size` | `-ts` | float | - | 0.2 | Test split fraction |
| `--seed` | `-s` | int | - | 42 | Random seed |
| `--extension_name` | `-en` | str | - | "" | Output file suffix |
| `--wanted_features` | `-wf` | int+ | - | None | Feature indices to plot |
| `--show_loss` | - | flag | - | False | Print training loss |

### Forward pass

| Phase | Component | Formula |
|-------|-----------|---------|
| Linear Transformation | $Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$ | • $A^{[l-1]} \in \mathbb{R}^{m \times n_{l-1}}$: Activations from previous layer<br>• $W^{[l]} \in \mathbb{R}^{n_{l-1} \times n_l}$: Weight matrix connecting layers<br>• $b^{[l]} \in \mathbb{R}^{1 \times n_l}$: Bias vector for layer $l$<br>• $Z^{[l]} \in \mathbb{R}^{m \times n_l}$: Pre-activation values (linear output) |
| Non-Linear Activation | $A^{[l]} = g^{[l]}(Z^{[l]})$ | • $g^{[l]}(\cdot)$: Activation function applied element-wise<br>• $A^{[l]} \in \mathbb{R}^{m \times n_l}$: Post-activation values (non-linear output)<br>• Common choices: ReLU, Sigmoid, Tanh |

### Backward pass

| Phase | Component | Formula |
|-------|-----------|---------|
| Weight Gradient | $\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} (A^{[l-1]})^\top \frac{\partial \mathcal{L}}{\partial Z^{[l]}}$ | • Matrix multiplication: $(n_{l-1} \times m) \times (m \times n_l) = (n_{l-1} \times n_l)$<br>• Factor $\frac{1}{m}$ averages gradients over mini-batch<br>• Add regularization: $+ \lambda_2 W^{[l]}$ (L2) or $+ \lambda_1 \text{sign}(W^{[l]})$ (L1) |
| Bias Gradient | $\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{m} \sum_{i=1}^m \frac{\partial \mathcal{L}}{\partial Z^{[l]}_i}$ | • Sum gradients across batch dimension (axis 0)<br>• Result shape: $(1 \times n_l)$ or $(n_l,)$<br>• Each bias unit receives gradient from all batch examples |
| Activation Gradient | $\frac{\partial \mathcal{L}}{\partial A^{[l-1]}} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}} (W^{[l]})^\top$ | • Matrix multiplication: $(m \times n_l) \times (n_l \times n_{l-1}) = (m \times n_{l-1})$<br>• Propagates error signal to previous layer<br>• Next: $\frac{\partial \mathcal{L}}{\partial Z^{[l-1]}} = \frac{\partial \mathcal{L}}{\partial A^{[l-1]}} \odot g'^{[l-1]}(Z^{[l-1]})$ |

## Key Implementation Notes

**Batch Processing:**
- All computations handle mini-batches of size $m$ simultaneously
- Gradients are averaged over the batch for stable updates

**Chain Rule Application:**
- Each layer's gradients depend on the gradients from the next layer
- The process flows backward from output to input layers

**Regularization Integration:**
- Weight gradients include regularization terms during the update step
- Bias terms typically are not regularized

**Memory Efficiency:**
- Forward pass caches $A^{[l-1]}$ and $Z^{[l]}$ for backward pass
- Activation gradients $\frac{\partial \mathcal{L}}{\partial A^{[l-1]}}$ are computed on-the-fly

## Mathematical foundation

### Activation functions

| Concept                   | Link                                                                                                     | ***TL;DR***                                                       |
| ------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| ReLU                      | [Wikipedia: ReLU](https://en.wikipedia.org/wiki/Rectifier_%28neural_networks%29)                         | Zeroes out negatives; fast + great default for deep nets.   |
| Sigmoid                   | [Wikipedia: Logistic function](https://en.wikipedia.org/wiki/Logistic_function)                          | Squashes to $[0,1]$; can saturate/vanish gradients.         |
| Tanh                      | [Wikipedia: Hyperbolic tangent](https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent)   | Zero-centered sigmoid; still can saturate.                  |
| Softmax                   | [Wikipedia: Softmax function](https://en.wikipedia.org/wiki/Softmax_function)                            | Converts logits to a probability distribution over classes. |

### Loss estimators

| Concept                   | Link                                                                                                     | ***TL;DR***                                                       |
| ------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Cross-Entropy Loss        | [Wikipedia: Cross entropy](https://en.wikipedia.org/wiki/Cross_entropy)                                  | Measures distance between predicted and true distributions. |
| MSE                       | [Wikipedia: Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)                        | Penalizes squared errors; smooth, standard for regression.  |
| MAE                       | [Wikipedia: Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)                      | Robust to outliers vs MSE; L1 error measure.                |
| RMSE                      | [Wikipedia: Root-mean-square deviation](https://en.wikipedia.org/wiki/Root-mean-square_deviation)        | Error in original units; $\sqrt{\text{MSE}}$.               |

### Regularization techniques

| Concept                   | Link                                                                                                     | ***TL;DR***                                                       |
| ------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| L1 Regularization (Lasso) | [Wikipedia: Lasso (statistics)](https://en.wikipedia.org/wiki/Lasso_%28statistics%29)                    | Encourages sparsity; can zero out weights.                  |
| L2 Regularization (Ridge) | [Wikipedia: Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization)              | Shrinks weights smoothly; combats overfitting.              |

### Optimize weights training

| Concept                   | Link                                                                                                     | ***TL;DR***                                                       |
| ------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Xavier/Glorot Init        | [PyTorch: xavier\_uniform](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_)   | Balanced variance for tanh/sigmoid nets.                    |
| He/Kaiming Init           | [PyTorch: kaiming\_uniform](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_) | Keeps ReLU activations well-scaled in deep nets.            |
| Learning Rate Decay       | [Wikipedia: Learning rate](https://en.wikipedia.org/wiki/Learning_rate)                                  | Gradually reduces step size to stabilize late training.     |
| Gradient Clipping         | [Wikipedia: Exploding gradient problem](https://en.wikipedia.org/wiki/Exploding_gradient_problem)        | Caps large gradients to avoid instability.                  |
