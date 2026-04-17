"""Produce a stratified train/val/test split JSON from the raw data directory.

Usage
-----
    python scripts/prepare_splits.py \
        --data-root ../Data/ALL \
        --out configs/splits.json \
        --ratios 0.7 0.15 0.15 \
        --seed 42

The output JSON has the form:

    {
      "train": {"Amhara": ["A (1).mp4", ...], "Oromo": [...], ...},
      "val":   {...},
      "test":  {...},
      "meta":  {"ratios": [...], "seed": 42, "counts": {...}}
    }
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# This mapping mirrors configs/base.yaml `data.class_dirs`.
DEFAULT_CLASS_DIRS = {
    "Amhara":   "data set Amhara",
    "Oromo":    "data set oromo",
    "Woliyta":  "data set Woliyta",
    "Gurageya": "data set Gurageya",
    "Tigriga":  "data set Tigriga",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def list_videos(folder: Path) -> list[str]:
    return sorted([p.name for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS])


def stratified_split(items: list[str], ratios: tuple[float, float, float], rng: random.Random
                     ) -> tuple[list[str], list[str], list[str]]:
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    # Ensure the test split gets the remainder so we never drop clips.
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path,
                    help="Folder containing the 5 class subdirectories.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination JSON path.")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"Ratios must sum to 1, got {args.ratios}")

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Data root not found: {data_root}")

    rng = random.Random(args.seed)
    splits = {"train": {}, "val": {}, "test": {}, "meta": {
        "ratios": list(args.ratios),
        "seed": args.seed,
        "counts": {},
        "data_root": str(data_root),
    }}

    total_counts = {"train": 0, "val": 0, "test": 0}
    for class_name, folder_name in DEFAULT_CLASS_DIRS.items():
        folder = data_root / folder_name
        if not folder.is_dir():
            raise SystemExit(f"Missing class folder: {folder}")
        files = list_videos(folder)
        if not files:
            raise SystemExit(f"No videos in {folder}")
        tr, va, te = stratified_split(files, tuple(args.ratios), rng)
        splits["train"][class_name] = tr
        splits["val"][class_name] = va
        splits["test"][class_name] = te
        splits["meta"]["counts"][class_name] = {
            "total": len(files), "train": len(tr), "val": len(va), "test": len(te)
        }
        for k, v in (("train", tr), ("val", va), ("test", te)):
            total_counts[k] += len(v)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.out}")
    print(f"  train={total_counts['train']}  val={total_counts['val']}  test={total_counts['test']}")
    for cls, c in splits["meta"]["counts"].items():
        print(f"  {cls:10s} tr={c['train']:4d}  va={c['val']:4d}  te={c['test']:4d}  (total {c['total']})")


if __name__ == "__main__":
    main()
