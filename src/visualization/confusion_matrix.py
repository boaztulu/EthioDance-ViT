"""Publication-quality confusion matrix with row-normalized percentages."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .plot_style import savefig_dual


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str],
    out_path: str | Path,
    *,
    normalize: str = "row",
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> None:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)

    if normalize == "row":
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_sum * 100.0
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_norm[i, j]:.1f}%\n({int(cm[i, j])})"
        heat = cm_norm
        fmt = ""
        cbar_label = "Row %"
    else:
        annot = cm.astype(int)
        heat = cm
        fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(1.2 * len(class_names) + 2, 1.0 * len(class_names) + 1.8))
    sns.heatmap(
        heat,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": cbar_label},
        square=True,
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    savefig_dual(fig, out_path)
    plt.close(fig)
