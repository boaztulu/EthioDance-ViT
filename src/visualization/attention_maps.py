"""Spatiotemporal attention visualization for TimeSformer (attention rollout).

TimeSformer uses *divided space-time attention*: each encoder block has both a
temporal attention sub-layer and a spatial attention sub-layer. We collect the
spatial attention maps from every block, apply Abnar & Zuidema (2020) style
rollout with residual accounting, then reshape the CLS-to-patch attention of
the final layer back to a (H_patches × W_patches) map per frame and overlay it
on the original frames.

This is what lets a paper say "the model focuses on the shoulders for Eskista"
— the overlay shows exactly which spatial regions drive the decision at each
sampled frame.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .plot_style import savefig_dual


class _SpatialAttnCollector:
    """Hooks HF TimeSformer spatial-attention modules to capture weights."""

    def __init__(self) -> None:
        self.spatial_attentions: list[torch.Tensor] = []
        self._hooks: list = []

    def attach(self, backbone) -> None:
        """``backbone`` is ``TimesformerForVideoClassification.timesformer``."""
        for block in backbone.encoder.layer:
            attn = block.attention.attention  # TimesformerSelfAttention (spatial)

            def _hook(_m, _inp, out, store=self.spatial_attentions):
                # HF returns (context, attn_probs) when output_attentions=True.
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    store.append(out[1].detach())

            self._hooks.append(attn.register_forward_hook(_hook))

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _rollout(attentions: list[torch.Tensor]) -> torch.Tensor:
    """Attention rollout over a list of (B, heads, N, N) spatial attentions."""
    # Fuse heads by mean, add identity to account for residual, normalize rows,
    # and multiply successively.
    fused = []
    for a in attentions:
        a = a.mean(dim=1)                           # (B, N, N)
        I = torch.eye(a.size(-1), device=a.device)  # noqa: E741
        a = a + I
        a = a / a.sum(dim=-1, keepdim=True)
        fused.append(a)
    out = fused[0]
    for a in fused[1:]:
        out = a @ out
    return out  # (B, N, N)


@torch.no_grad()
def _forward_with_attn(model, pixel_values: torch.Tensor):
    """Run a forward pass collecting per-block spatial attentions."""
    collector = _SpatialAttnCollector()
    collector.attach(model.backbone.timesformer)
    try:
        # output_attentions=True triggers the hook payload to include attn probs.
        outputs = model.backbone(pixel_values=pixel_values, output_attentions=True)
    finally:
        collector.detach()
    return outputs.logits, collector.spatial_attentions


def _overlay(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a viridis heatmap onto an RGB frame (both 0..1)."""
    cmap = plt.get_cmap("jet")
    hm = cmap(heatmap)[..., :3]
    return (1 - alpha) * frame + alpha * hm


def _denormalize(clip: torch.Tensor, mean, std) -> np.ndarray:
    m = torch.tensor(mean).view(1, -1, 1, 1)
    s = torch.tensor(std).view(1, -1, 1, 1)
    img = clip.cpu() * s + m
    return img.clamp(0, 1).permute(0, 2, 3, 1).numpy()  # (T, H, W, 3)


def visualize_attention(
    model,
    clip: torch.Tensor,          # (T, C, H, W), already normalized
    class_names: list[str],
    out_path: str | Path,
    *,
    target_label: int | None = None,
    mean=(0.45, 0.45, 0.45),
    std=(0.225, 0.225, 0.225),
    device: torch.device | None = None,
    title: str | None = None,
) -> int:
    """Generate a single grid figure of frames with attention overlay.

    Returns the predicted class index.
    """
    device = device or next(model.parameters()).device
    model.eval()
    x = clip.unsqueeze(0).to(device)    # (1, T, C, H, W)
    logits, attns = _forward_with_attn(model, x)
    pred = int(logits.argmax(dim=-1).item())

    if not attns:
        raise RuntimeError("No spatial attentions captured — check the hook target.")

    # attns: list of (B, heads, N, N) where N = 1 (CLS) + num_patches_per_frame.
    # TimeSformer spatial attention is computed per-frame, so batch dim is
    # already B*T. Reshape accordingly.
    B = x.size(0)
    T = x.size(1)
    rolled = _rollout(attns)            # (B*T, N, N)
    cls_to_patches = rolled[:, 0, 1:]   # (B*T, num_patches)
    num_patches = cls_to_patches.size(-1)
    grid = int(round(num_patches ** 0.5))
    maps = cls_to_patches.view(B, T, grid, grid)

    # Normalize each frame's map to [0, 1] for display.
    maps = maps - maps.amin(dim=(-2, -1), keepdim=True)
    maps = maps / maps.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    # Upsample to frame resolution.
    H, W = x.shape[-2], x.shape[-1]
    maps_up = F.interpolate(maps[0].unsqueeze(1), size=(H, W), mode="bilinear",
                            align_corners=False).squeeze(1).cpu().numpy()  # (T, H, W)

    frames = _denormalize(clip, mean, std)  # (T, H, W, 3)

    cols = min(T, 8)
    rows = int(np.ceil(T / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(1.9 * cols, 2.1 * rows * 2))
    if rows * 2 == 1:
        axes = np.asarray(axes).reshape(1, -1)

    for t in range(T):
        r = (t // cols) * 2
        c = t % cols
        ax_top = axes[r, c]
        ax_bot = axes[r + 1, c]
        ax_top.imshow(frames[t])
        ax_top.set_title(f"t={t}", fontsize=8)
        ax_top.axis("off")
        ax_bot.imshow(_overlay(frames[t], maps_up[t]))
        ax_bot.axis("off")

    # Blank any leftover panels.
    for k in range(T, rows * cols):
        r = (k // cols) * 2
        c = k % cols
        axes[r, c].axis("off")
        axes[r + 1, c].axis("off")

    if title is None:
        pred_name = class_names[pred]
        title = f"Attention rollout → predicted: {pred_name}"
        if target_label is not None:
            title += f" (truth: {class_names[target_label]})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    savefig_dual(fig, out_path)
    plt.close(fig)
    return pred
