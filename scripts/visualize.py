"""Generate all paper figures from a trained checkpoint.

Produces, under <run-dir>/figures/:
  - confusion_matrix.{pdf,png}
  - training_curves.{pdf,png}        (requires history.json in <run-dir>)
  - embeddings.{pdf,png}             (t-SNE + UMAP of test-set features)
  - attention/<class>_<id>.{pdf,png} (one example per class)
  - classification_report.txt

Usage
-----
    python scripts/visualize.py --config configs/hipergator.yaml \
        --checkpoint ../experiments/run_20260417_.../checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_datasets, build_transform                 # noqa: E402
from src.models import build_timesformer                             # noqa: E402
from src.utils import get_logger, load_config, set_seed, setup_logging  # noqa: E402
from src.utils.checkpoint import load_checkpoint                     # noqa: E402
from src.utils.metrics import compute_classification_report          # noqa: E402
from src.visualization import (                                      # noqa: E402
    plot_confusion_matrix, plot_embeddings, plot_training_curves,
    set_paper_style, visualize_attention,
)
from src.visualization.embeddings import extract_dataset_features    # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Defaults to <checkpoint dir>/../figures")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--attention-examples-per-class", type=int, default=1)
    ap.add_argument("-o", "--override", action="append", default=[])
    return ap.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        logits = model(pixel_values)
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        all_targets.extend(batch["labels"].numpy().tolist())
    return np.asarray(all_targets), np.asarray(all_preds)


def main() -> None:
    args = parse_args()
    set_paper_style()

    cfg = load_config(args.config, overrides=args.override)
    run_dir = args.checkpoint.resolve().parent.parent
    fig_dir = (args.output_dir or (run_dir / "figures")).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    log = get_logger("visualize")
    set_seed(int(cfg["experiment"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | figures -> %s", device, fig_dir)

    # -- Model --
    model = build_timesformer(cfg).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    log.info("Loaded checkpoint %s", args.checkpoint)

    # -- Data --
    transforms_by_split = {s: build_transform(cfg, s) for s in ("train", "val", "test")}
    datasets = build_datasets(cfg, transforms_by_split, debug=False)
    class_names = list(cfg["data"]["class_dirs"].keys())

    eval_ds = datasets[args.split]
    loader = DataLoader(
        eval_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["eval"]["num_workers"],
        pin_memory=True,
    )

    # -- 1. Confusion matrix + classification report --
    y_true, y_pred = collect_predictions(model, loader, device)
    plot_confusion_matrix(
        y_true, y_pred, class_names, fig_dir / "confusion_matrix",
        normalize="row", title="TimeSformer — Ethiopian Dance Classification",
    )
    report = compute_classification_report(y_true, y_pred, class_names)
    (fig_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    log.info("Confusion matrix + report saved.")

    # -- 2. Training curves (requires history.json) --
    hist_path = run_dir / "history.json"
    if hist_path.exists():
        with open(hist_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        plot_training_curves(history, fig_dir / "training_curves")
        log.info("Training curves saved.")
    else:
        log.warning("history.json not found at %s — skipping training curves.", hist_path)

    # -- 3. t-SNE / UMAP of penultimate features --
    feats, labels = extract_dataset_features(model, loader, device)
    plot_embeddings(feats, labels, class_names, fig_dir / "embeddings", method="both")
    log.info("Feature embeddings saved.")

    # -- 4. Attention overlays (one per class) --
    attn_dir = fig_dir / "attention"
    attn_dir.mkdir(exist_ok=True)
    chosen: dict[int, int] = {}
    for i, rec in enumerate(eval_ds.records):
        if rec.label not in chosen:
            chosen[rec.label] = i
        if len(chosen) == len(class_names):
            break

    for class_idx, ds_idx in chosen.items():
        sample = eval_ds[ds_idx]
        clip = sample["pixel_values"]
        target = int(sample["labels"])
        out_name = attn_dir / f"{class_names[class_idx]}_{ds_idx}"
        try:
            visualize_attention(
                model, clip, class_names, out_name,
                target_label=target, device=device,
                mean=cfg["data"]["augmentation"]["mean"],
                std=cfg["data"]["augmentation"]["std"],
            )
            log.info("Attention map saved: %s", out_name.name)
        except Exception as exc:
            log.warning("Attention viz failed for %s: %s", class_names[class_idx], exc)

    log.info("All figures saved under %s", fig_dir)


if __name__ == "__main__":
    main()
