"""2-D projection of penultimate features via t-SNE and/or UMAP.

Extract CLS features via ``TimeSformerClassifier.extract_features`` on a data
loader, reduce them, and render one scatter per class.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from .plot_style import savefig_dual


@torch.no_grad()
def extract_dataset_features(model, loader: DataLoader, device: torch.device
                             ) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    feats, labels = [], []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        f = model.extract_features(pixel_values).cpu().numpy()
        feats.append(f)
        labels.append(batch["labels"].numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def _scatter(ax, X2: np.ndarray, labels: np.ndarray, class_names: list[str], title: str) -> None:
    palette = plt.get_cmap("tab10")
    for c, name in enumerate(class_names):
        mask = labels == c
        if not mask.any():
            continue
        ax.scatter(
            X2[mask, 0], X2[mask, 1],
            s=22, alpha=0.85, edgecolor="white", linewidth=0.3,
            color=palette(c % 10), label=name,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="best", fontsize=8)


def plot_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    out_path: str | Path,
    *,
    method: str = "both",  # "tsne" | "umap" | "both"
    seed: int = 42,
) -> None:
    runs: list[tuple[str, np.ndarray]] = []
    if method in ("tsne", "both"):
        perplexity = float(min(30, max(5, (len(features) - 1) // 3)))
        ts = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  learning_rate="auto", random_state=seed)
        runs.append(("t-SNE", ts.fit_transform(features)))
    if method in ("umap", "both"):
        try:
            import umap  # type: ignore
            um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed)
            runs.append(("UMAP", um.fit_transform(features)))
        except Exception:
            if method == "umap":
                raise
            # Silent fallback if UMAP not installed and method=="both".

    ncols = len(runs)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.0), squeeze=False)
    for ax, (name, X2) in zip(axes[0], runs):
        _scatter(ax, X2, labels, class_names, name)
    fig.tight_layout()
    savefig_dual(fig, out_path)
    plt.close(fig)
