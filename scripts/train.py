"""Entry point: train TimeSformer on Ethiopian dance videos.

Examples
--------
    # Local smoke test
    python scripts/train.py --config configs/local.yaml --debug

    # Full HiPerGator run, overriding a field on the CLI
    python scripts/train.py --config configs/hipergator.yaml \
        --output-dir ../experiments/run_$(date +%Y%m%d_%H%M%S) \
        -o train.epochs=100 -o optim.lr=3e-5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_datasets, build_transform, build_sampler  # noqa: E402
from src.engine import EarlyStopping, Trainer                        # noqa: E402
from src.losses import build_loss                                    # noqa: E402
from src.models import build_timesformer, param_groups               # noqa: E402
from src.utils import (                                              # noqa: E402
    RequeueHandler, get_logger, load_config, save_config, set_seed, setup_logging,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mode", choices=["local", "hipergator"], default=None,
                    help="Selects configs/local.yaml or configs/hipergator.yaml when --config is omitted.")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Run dir; default: <experiment.output_root>/<tag>_<timestamp>.")
    ap.add_argument("--debug", action="store_true", help="Force debug.enabled=true.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last.pth in the chosen output dir (auto-detected by default).")
    ap.add_argument("-o", "--override", action="append", default=[],
                    help="Dotted override, e.g. -o train.batch_size=4")
    return ap.parse_args()


def build_loaders(cfg: dict, debug: bool):
    transforms_by_split = {s: build_transform(cfg, s) for s in ("train", "val", "test")}
    datasets = build_datasets(cfg, transforms_by_split, debug=debug)

    tcfg = cfg["train"]
    ecfg = cfg["eval"]

    sampler = build_sampler(datasets["train"].labels, tcfg.get("sampler", "shuffle"))
    train_loader = DataLoader(
        datasets["train"],
        batch_size=tcfg["batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=tcfg["num_workers"],
        pin_memory=tcfg["pin_memory"],
        drop_last=True,
        persistent_workers=tcfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=ecfg["batch_size"],
        shuffle=False,
        num_workers=ecfg["num_workers"],
        pin_memory=tcfg["pin_memory"],
        drop_last=False,
        persistent_workers=ecfg["num_workers"] > 0,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=ecfg["batch_size"],
        shuffle=False,
        num_workers=ecfg["num_workers"],
        pin_memory=tcfg["pin_memory"],
        drop_last=False,
        persistent_workers=ecfg["num_workers"] > 0,
    )
    return datasets, (train_loader, val_loader, test_loader)


def main() -> None:
    args = parse_args()

    cfg_path = args.config
    if cfg_path is None and args.mode:
        cfg_path = REPO_ROOT / "configs" / f"{args.mode}.yaml"
    if cfg_path is None:
        raise SystemExit("Either --config or --mode must be supplied.")

    cfg = load_config(cfg_path, overrides=args.override)
    if args.debug:
        cfg.setdefault("debug", {})["enabled"] = True

    # Output dir (outside the repo).
    if args.output_dir is not None:
        out = args.output_dir
    else:
        root = Path(cfg["experiment"]["output_root"])
        tag = cfg["experiment"].get("tag", "run")
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = root / f"{tag}_{ts}"
    out = out.resolve()
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    setup_logging(out)
    log = get_logger("train")
    log.info("Output dir: %s", out)
    log.info("Config: %s", cfg_path)
    save_config(cfg, out / "resolved_config.yaml")

    set_seed(int(cfg["experiment"]["seed"]))

    # Build data.
    debug = bool(cfg.get("debug", {}).get("enabled", False))
    datasets, (train_loader, val_loader, test_loader) = build_loaders(cfg, debug)
    class_names = list(cfg["data"]["class_dirs"].keys())
    log.info("Dataset sizes | train=%d val=%d test=%d",
             len(datasets["train"]), len(datasets["val"]), len(datasets["test"]))

    # Build model / loss / optim.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = build_timesformer(cfg)
    loss_fn = build_loss(cfg, datasets["train"].labels)

    groups = param_groups(
        model,
        base_lr=float(cfg["optim"]["lr"]),
        head_lr_mult=float(cfg["optim"]["head_lr_mult"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg["optim"]["betas"]))

    # Logging: TensorBoard.
    tb = None
    if cfg["logging"].get("tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=str(out / "tb"))
        except Exception as exc:
            log.warning("TensorBoard unavailable: %s", exc)

    early = None
    if cfg["train"]["early_stopping"]["enabled"]:
        monitor = cfg["train"]["early_stopping"]["metric"]
        mode = "min" if monitor.endswith("loss") else "max"
        early = EarlyStopping(
            patience=int(cfg["train"]["early_stopping"]["patience"]),
            min_delta=float(cfg["train"]["early_stopping"]["min_delta"]),
            mode=mode,
        )

    requeue = RequeueHandler()
    requeue.install()

    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=out,
        class_names=class_names,
        tb_writer=tb,
        early_stopper=early,
        requeue=requeue,
    )

    # Auto-resume if last.pth exists in the output dir.
    if args.resume or (out / "checkpoints" / "last.pth").exists():
        trainer.maybe_resume()

    history = trainer.fit()

    # Persist training history for visualize.py.
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history.__dict__, f, indent=2)

    # Final test evaluation using the best checkpoint.
    best = out / "checkpoints" / "best.pth"
    if best.exists():
        from src.utils.checkpoint import load_checkpoint
        load_checkpoint(best, model=model, map_location=device)
        log.info("Loaded best.pth for final test evaluation.")

    test_stats = trainer._eval_epoch(epoch=-1, loader=test_loader, tag="test")  # noqa: SLF001
    log.info("TEST | loss %.4f acc %.4f f1 %.4f",
             test_stats["loss"], test_stats["accuracy"], test_stats["f1_macro"])
    with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_stats, f, indent=2)

    if tb is not None:
        tb.close()

    log.info("Done. Artifacts in %s", out)


if __name__ == "__main__":
    main()
