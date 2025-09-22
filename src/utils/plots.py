from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


def normalize_colors(
    data: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> Normalize:
    """
    Normalize the colors for a given data range.

    Args:
        data (np.ndarray): The data to normalize.
        vmin (Optional[float], optional): Minimum value for normalization. Defaults to None.
        vmax (Optional[float], optional): Maximum value for normalization. Defaults to None.

    Returns:
        Normalize: A Normalize instance for the given range.
    """
    return Normalize(
        vmin=vmin if vmin is not None else np.min(data),
        vmax=vmax if vmax is not None else np.max(data)
    )


def plot_heatmap(
    data: np.ndarray,
    x_labels: Optional[np.ndarray] = None,
    y_labels: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    color_map: str = "viridis",
    color_bar_label: str = "",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save_path: Optional[str] = None,
) -> tuple[Figure, Axes]:
    """
    Plot a heatmap from a 2D numpy array.

    Args:
        data (np.ndarray):
            2D array of data to plot.
        x_labels (Optional[np.ndarray], optional):
            Labels for the x-axis. Defaults to None.
        y_labels (Optional[np.ndarray], optional):
            Labels for the y-axis. Defaults to None.
        ax (Optional[Axes], optional):
            Matplotlib Axes object to plot on.
            If None, a new figure and axes are created. Defaults to None.
        fig (Optional[Figure], optional):
            Matplotlib Figure object to plot on.
            If None, a new figure and axes are created. Defaults to None.
        color_map (str, optional):
            Colormap to use for the heatmap. Defaults to "viridis".
        color_bar_label (str, optional):
            Label for the color bar. Defaults to "".
        title (str, optional):
            Title of the plot. Defaults to "".
        xlabel (str, optional):
            Label for the x-axis. Defaults to "".
        ylabel (str, optional):
            Label for the y-axis. Defaults to "".
        save_path (Optional[str], optional):
            If provided, saves the plot to the given path. Defaults to None.

    Returns:
        tuple[Figure, Axes]:
            The Matplotlib Figure and Axes objects containing the heatmap.
    """
    if ax is None or fig is None:
        fig, ax = plt.subplots()

    cax = ax.imshow(data, cmap=color_map, aspect='auto')

    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(color_bar_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig, ax


def plot_function_evaluation(
    x: np.ndarray,
    y: np.ndarray,
    data: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    color_map: str = "viridis",
    color_bar_label: str = "",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    save_path: Optional[str] = None,
    *,
    mode: str = "auto",                # "auto" | "probs" | "labels"
    class_names: Optional[Sequence[str]] = None,
    show_threshold: bool = True        # for binary sigmoid
) -> tuple[Figure, Axes]:
    """
    Robust plotting for MLP outputs.

    - Multiclass classification: y.shape == (N, C) with C > 1
        -> scatter: x vs predicted class index; color = max confidence in [0,1]
    - Regression or binary output: y shape (N,) or (N,1)
        -> line plot of y vs x (sorted by x)

    Notes:
    - If x is higher-dimensional, it will be flattened.
    - If y is (N,1) it is squeezed to (N,).
    - If `save_path` is provided, the figure is saved and closed.

    - softmax (N,C):
        mode="probs"  -> plot p(class k|x) lines for each k
        mode="labels" -> scatter argmax class vs x with color = confidence
        mode="auto"   -> "probs" if C<=5 else "labels"

    - sigmoid/regression:
        (N,) or (N,1) -> line plot; if ~binary in [0,1] and show_threshold, draw y=0.5
    """
    dim = 2  # for 2D plots
    new_fig = False
    if ax is None or fig is None:
        fig, ax = plt.subplots()
        new_fig = True

    # Normalize x
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.reshape(-1)

    # Normalize y
    y = np.asarray(y)
    if y.ndim == dim and y.shape[1] == 1:
        y = y[:, 0]

    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or Inf values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or Inf values.")
    if class_names is None:
        class_names = []

    # Sort by x for nicer curves
    order = np.argsort(x)
    x_ord = x[order]

    if y.ndim == dim and y.shape[1] > 1:
        # Multiclass softmax
        C = y.shape[1]
        y_ord = y[order]

        chosen_mode = mode
        if mode == "auto":
            chosen_mode = "probs" if C <= len(class_names) else "labels"

        if chosen_mode == "probs":
            #############################################
            # Line plot of all class probabilities vs x #
            #############################################
            # line plot of all class probabilities vs x
            if class_names and len(class_names) != C:
                labels = [f"class {k}" for k in range(C)]
            else:
                labels = class_names
            # plot each class probability
            for k in range(C):
                ax.plot(x_ord, y_ord[:, k], label=labels[k])
            # add horizontal line at y=1/C
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc="best")
            ax.set_ylabel(ylabel or "probability")
        else:
            ###############################################################
            # Scatter plot of predicted class vs x, colored by confidence #
            ###############################################################
            # scatter plot of predicted class vs x, colored by confidence
            cls = np.argmax(y_ord, axis=1)
            conf = np.max(y_ord, axis=1)
            # scatter plot
            sc = ax.scatter(x_ord, cls, c=conf, cmap=color_map, s=18, edgecolors="none")
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(color_bar_label or "confidence")
            # determine unique classes
            uniq = np.unique(cls)
            if class_names and len(class_names) == C:
                ax.set_yticks(uniq)
                ax.set_yticklabels([str(class_names[i]) for i in uniq])
            else:
                ax.set_yticks(uniq)
            ax.set_ylabel(ylabel or "class")

    else:
        ######################################################
        # Line plot of y vs x (regression or binary sigmoid) #
        ######################################################
        # line plot of y vs x (regression or binary sigmoid)
        y_ord = y[order]
        ax.plot(x_ord, y_ord, linewidth=1.7)
        # add horizontal line at y=0.5 if binary sigmoid
        if show_threshold and np.all(y_ord >= 0.0) and np.all(y_ord <= 1.0):
            ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.7)
            ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(ylabel or "output")

    # Color bar for additional data if provided
    if data is not None:
        arr = np.asarray(data)
        if arr.size > 0 and np.all(np.isfinite(arr)):
            norm = Normalize(vmin=np.nanmin(arr), vmax=np.nanmax(arr))
            sm = ScalarMappable(norm=norm, cmap=color_map)
            sm.set_array(arr)
            cbar2 = fig.colorbar(sm, ax=ax)
            cbar2.set_label(color_bar_label or "scale")

    ax.set_title(title)
    ax.set_xlabel(xlabel or "x")
    fig.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        if new_fig:
            plt.close(fig)

    return fig, ax
