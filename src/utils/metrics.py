"""Classification metrics helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class MetricTracker:
    """Accumulates per-batch predictions and computes aggregate metrics."""

    preds: list[int] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float | None = None) -> None:
        preds = logits.detach().argmax(dim=-1).cpu().numpy().tolist()
        self.preds.extend(preds)
        self.targets.extend(targets.detach().cpu().numpy().tolist())
        if loss is not None:
            self.losses.append(float(loss))

    def reset(self) -> None:
        self.preds.clear()
        self.targets.clear()
        self.losses.clear()

    def summary(self) -> dict:
        if not self.targets:
            return {"accuracy": 0.0, "f1_macro": 0.0, "precision_macro": 0.0,
                    "recall_macro": 0.0, "loss": 0.0}
        y_true = np.asarray(self.targets)
        y_pred = np.asarray(self.preds)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "loss": float(np.mean(self.losses)) if self.losses else 0.0,
        }

    def confusion(self, num_classes: int) -> np.ndarray:
        labels = list(range(num_classes))
        return confusion_matrix(self.targets, self.preds, labels=labels)


def compute_classification_report(
    targets: Sequence[int], preds: Sequence[int], class_names: Sequence[str]
) -> str:
    return classification_report(
        targets, preds, target_names=list(class_names), digits=4, zero_division=0
    )
