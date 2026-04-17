"""TimeSformer builder with transfer-learning friendly freezing.

We use the HuggingFace implementation of TimeSformer (Bertasius et al., 2021)
loaded from the Kinetics-400 checkpoint. The classification head is replaced
with one sized to our number of dance classes, and (by default) all backbone
parameters are frozen except the last N transformer blocks and the head.

A small wrapper exposes two things the trainer needs:
  * ``forward(pixel_values) -> logits`` with input shape (B, T, C, H, W).
  * ``extract_features(pixel_values) -> (B, D)`` for t-SNE / UMAP plots.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn as nn
from transformers import TimesformerForVideoClassification, TimesformerConfig

log = logging.getLogger(__name__)


class TimeSformerClassifier(nn.Module):
    """Thin wrapper around HF's TimesformerForVideoClassification."""

    def __init__(self, hf_model: TimesformerForVideoClassification, dropout: float = 0.0) -> None:
        super().__init__()
        self.backbone = hf_model
        # Inject dropout before the classifier if requested.
        if dropout > 0.0:
            hidden = hf_model.config.hidden_size
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden, hf_model.config.num_labels),
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # HF expects (B, T, C, H, W)
        out = self.backbone(pixel_values=pixel_values)
        return out.logits

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return the pooled CLS feature used by the classifier (B, D)."""
        bb = self.backbone.timesformer
        outputs = bb(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
        # HF TimeSformer returns last_hidden_state of shape (B, 1+T*N_patches, D).
        # The [CLS] token representation is element 0.
        cls = outputs.last_hidden_state[:, 0]
        return cls


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_timesformer(cfg: dict) -> TimeSformerClassifier:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    name = model_cfg["name"]
    num_classes = int(data_cfg["num_classes"])
    num_frames = int(data_cfg["num_frames"])
    image_size = int(data_cfg["frame_size"])

    log.info("Loading %s (num_frames=%d, image_size=%d, num_classes=%d)",
             name, num_frames, image_size, num_classes)

    # Load pretrained config, override num_frames/image_size/num_labels so the
    # position embeddings and head are re-sized correctly.
    hf_cfg = TimesformerConfig.from_pretrained(name)
    hf_cfg.num_frames = num_frames
    hf_cfg.image_size = image_size
    hf_cfg.num_labels = num_classes
    hf_cfg.id2label = {i: f"class_{i}" for i in range(num_classes)}
    hf_cfg.label2id = {v: k for k, v in hf_cfg.id2label.items()}

    hf_model = TimesformerForVideoClassification.from_pretrained(
        name,
        config=hf_cfg,
        ignore_mismatched_sizes=True,  # new head + possibly resized position embs
    )
    model = TimeSformerClassifier(hf_model, dropout=float(model_cfg.get("dropout", 0.0)))

    if model_cfg.get("freeze_backbone", True):
        freeze_backbone(model, last_n_blocks_unfrozen=int(model_cfg.get("last_n_blocks_unfrozen", 0)))
    return model


def freeze_backbone(model: TimeSformerClassifier, *, last_n_blocks_unfrozen: int = 0) -> None:
    """Freeze all backbone params except the last N transformer encoder blocks.

    The classification head is always left trainable.
    """
    bb = model.backbone.timesformer
    for p in bb.parameters():
        p.requires_grad = False

    # Unfreeze last N encoder layers.
    layers = bb.encoder.layer
    if last_n_blocks_unfrozen > 0:
        for block in layers[-last_n_blocks_unfrozen:]:
            for p in block.parameters():
                p.requires_grad = True
        # The layer-norm after the encoder is shared with the head's view of features.
        for p in bb.layernorm.parameters():
            p.requires_grad = True

    # Classifier is always trainable.
    for p in model.backbone.classifier.parameters():
        p.requires_grad = True

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Backbone frozen. Trainable params: %.2fM / %.2fM (%.1f%%)",
             n_train / 1e6, n_total / 1e6, 100.0 * n_train / n_total)


def param_groups(model: TimeSformerClassifier, *, base_lr: float, head_lr_mult: float,
                 weight_decay: float) -> list[dict]:
    """Return optimizer param groups with a higher LR for the classifier head."""
    head_params = list(model.backbone.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return [
        {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": [p for p in head_params if p.requires_grad],
         "lr": base_lr * head_lr_mult, "weight_decay": weight_decay},
    ]
