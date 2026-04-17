"""Class-Balanced Focal Loss (Cui et al., CVPR 2019).

For each class c with n_c training samples we compute an effective number

    E_c = (1 - beta^{n_c}) / (1 - beta)

and set the per-class weight to alpha_c = 1 / E_c, normalized so that it sums
to num_classes. Those weights are then multiplied into the focal loss

    FL(p_t) = - alpha_c * (1 - p_t)^gamma * log(p_t)

Soft-label support: if ``targets`` is 2-D (from MixUp) we compute the loss
against the soft target via the standard cross-entropy-with-soft-labels trick
while keeping the focal modulation per sample.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_number_weights(class_counts: list[int], beta: float) -> torch.Tensor:
    counts = np.asarray(class_counts, dtype=np.float64)
    effective = 1.0 - np.power(beta, counts)
    # Avoid div by zero for empty classes.
    effective = np.where(effective > 0, effective, 1.0)
    weights = (1.0 - beta) / effective
    weights = weights / weights.sum() * len(counts)
    return torch.as_tensor(weights, dtype=torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 1:
            return self._hard(logits, targets)
        return self._soft(logits, targets)

    def _hard(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        t_onehot = F.one_hot(targets, logits.size(-1)).float()
        if self.label_smoothing > 0:
            t_onehot = t_onehot * (1 - self.label_smoothing) + self.label_smoothing / logits.size(-1)
        # Focal modulation with per-sample p_t.
        p_t = (probs * t_onehot).sum(dim=-1).clamp_min(1e-8)
        alpha_t = (self.class_weights[targets])
        focal = alpha_t * (1.0 - p_t).pow(self.gamma) * (-p_t.log())
        return focal.mean()

    def _soft(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: (B, C) probabilities (e.g. from MixUp).
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        p_t = (probs * targets).sum(dim=-1).clamp_min(1e-8)
        # Per-sample alpha = weighted average of class weights by target mass.
        alpha_t = (self.class_weights.unsqueeze(0) * targets).sum(dim=-1)
        # Soft CE with focal modulation.
        ce = -(targets * log_probs).sum(dim=-1)
        focal = alpha_t * (1.0 - p_t).pow(self.gamma) * ce
        return focal.mean()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_loss(cfg: dict, train_labels: list[int]) -> nn.Module:
    loss_cfg = cfg["loss"]
    kind = loss_cfg["type"]
    num_classes = int(cfg["data"]["num_classes"])
    counts = [0] * num_classes
    for c, n in Counter(train_labels).items():
        counts[c] = n

    if kind == "ce":
        return nn.CrossEntropyLoss()

    if kind == "weighted_ce":
        weights = torch.tensor(
            [0.0 if n == 0 else (sum(counts) / (num_classes * n)) for n in counts],
            dtype=torch.float32,
        )
        return nn.CrossEntropyLoss(weight=weights)

    if kind == "class_balanced_focal":
        weights = effective_number_weights(counts, beta=float(loss_cfg.get("cb_beta", 0.9999)))
        return ClassBalancedFocalLoss(
            class_weights=weights,
            gamma=float(loss_cfg.get("focal_gamma", 2.0)),
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
        )

    raise ValueError(f"Unknown loss type: {kind!r}")
