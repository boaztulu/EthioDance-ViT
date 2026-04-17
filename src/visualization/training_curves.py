"""Training / validation loss and accuracy curves, two-panel layout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .plot_style import savefig_dual


def plot_training_curves(history: dict, out_path: str | Path) -> None:
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax_loss.plot(epochs, history.get("train_loss", []), label="Train", color="#1f77b4")
    if history.get("val_loss"):
        ax_loss.plot(epochs, history["val_loss"], label="Val", color="#d62728")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(frameon=False)

    ax_acc.plot(epochs, history.get("train_acc", []), label="Train", color="#1f77b4")
    if history.get("val_acc"):
        ax_acc.plot(epochs, history["val_acc"], label="Val", color="#d62728")
    if history.get("val_f1"):
        ax_acc.plot(epochs, history["val_f1"], label="Val F1", color="#2ca02c", linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy / F1")
    ax_acc.set_title("Accuracy / F1 (macro)")
    ax_acc.set_ylim(0, 1.02)
    ax_acc.legend(frameon=False)

    fig.tight_layout()
    savefig_dual(fig, out_path)
    plt.close(fig)
