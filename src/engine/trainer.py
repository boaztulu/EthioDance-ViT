"""Training loop with mixed precision, grad accumulation, and SLURM-safe state.

Design notes
------------
* The trainer owns `epoch`, `best_metric` and the optimizer/scheduler/scaler,
  so a single call to ``save_resumable()`` captures everything needed to pick
  up after a requeue.
* A `RequeueHandler` is polled once per step; when triggered, we save a
  resumable checkpoint and exit cleanly. SLURM re-runs the same script which
  loads ``last.pth`` and continues.
* MixUp produces soft labels; accuracy metrics use the argmax of the hard
  label, so during MixUp batches the training accuracy is computed against
  the original y_a (pre-mix), which matches common practice.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.mixup import VideoMixUp
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.metrics import MetricTracker
from ..utils.signals import RequeueHandler

log = logging.getLogger(__name__)


@dataclass
class EpochHistory:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        *,
        cfg: dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
        class_names: list[str],
        tb_writer=None,
        early_stopper=None,
        requeue: RequeueHandler | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.tb = tb_writer
        self.early = early_stopper
        self.requeue = requeue or RequeueHandler()

        tcfg = cfg["train"]
        self.epochs = int(tcfg["epochs"])
        self.grad_accum = max(1, int(tcfg.get("grad_accum_steps", 1)))
        self.grad_clip = float(tcfg.get("grad_clip_norm", 0.0))
        self.amp = bool(tcfg.get("mixed_precision", False)) and device.type == "cuda"
        self.log_every = int(cfg["logging"].get("log_every_n_steps", 10))
        self.save_every = int(tcfg["checkpoint"].get("save_every_n_epochs", 5))

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.scheduler = self._build_scheduler()

        self.mixup = VideoMixUp(
            alpha=float(cfg["data"]["mixup"].get("alpha", 0.0)),
            num_classes=int(cfg["data"]["num_classes"]),
            enabled=bool(cfg["data"]["mixup"].get("enabled", False)),
        )

        self.history = EpochHistory()
        self.start_epoch = 0
        self.best_metric = -math.inf
        self.monitor = tcfg["early_stopping"]["metric"]
        self.monitor_mode = "min" if self.monitor.endswith("loss") else "max"

        self._debug = bool(cfg.get("debug", {}).get("enabled", False))
        self._max_train_batches = cfg.get("debug", {}).get("max_train_batches")
        self._max_val_batches = cfg.get("debug", {}).get("max_val_batches")

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    def _build_scheduler(self):
        scfg = self.cfg["scheduler"]
        if scfg["type"] == "none":
            return None
        warmup_epochs = int(scfg.get("warmup_epochs", 0))
        min_lr = float(scfg.get("min_lr", 0.0))
        total_epochs = self.epochs

        cosine = CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr
        )
        if warmup_epochs <= 0:
            return cosine

        def warmup_lambda(epoch: int) -> float:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        warmup = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
        return SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    # ------------------------------------------------------------------
    # Checkpoint paths
    # ------------------------------------------------------------------
    def _ckpt(self, name: str) -> Path:
        return self.output_dir / "checkpoints" / f"{name}.pth"

    def save_resumable(self, epoch: int) -> None:
        save_checkpoint(
            self._ckpt("last"),
            model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, scaler=self.scaler,
            epoch=epoch, best_metric=self.best_metric,
            extra={"history": self.history.__dict__},
        )

    def save_best(self, epoch: int) -> None:
        save_checkpoint(
            self._ckpt("best"),
            model=self.model, optimizer=self.optimizer,
            scheduler=self.scheduler, scaler=self.scaler,
            epoch=epoch, best_metric=self.best_metric,
            extra={"class_names": self.class_names},
        )

    def maybe_resume(self) -> None:
        last = self._ckpt("last")
        if last.exists():
            log.info("Resuming from %s", last)
            state = load_checkpoint(
                last, model=self.model, optimizer=self.optimizer,
                scheduler=self.scheduler, scaler=self.scaler,
                map_location=self.device,
            )
            self.start_epoch = int(state["epoch"]) + 1
            self.best_metric = float(state["best_metric"])
            hist = state.get("extra", {}).get("history")
            if hist:
                for k, v in hist.items():
                    setattr(self.history, k, list(v))

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        tracker = MetricTracker()
        self.optimizer.zero_grad(set_to_none=True)

        batches_per_epoch = len(self.train_loader)
        if self._debug and self._max_train_batches:
            batches_per_epoch = min(batches_per_epoch, int(self._max_train_batches))

        pbar = tqdm(self.train_loader, desc=f"train[{epoch}]", total=batches_per_epoch, leave=False)
        step = 0
        running = 0.0
        for i, batch in enumerate(pbar):
            if self._debug and self._max_train_batches and i >= int(self._max_train_batches):
                break

            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # MixUp: x' and soft targets. Keep hard labels for running metrics.
            hard_labels = labels
            mixed_x, soft_targets = self.mixup(pixel_values, labels)

            with torch.amp.autocast("cuda", enabled=self.amp):
                logits = self.model(mixed_x)
                loss = self.loss_fn(logits, soft_targets if self.mixup.enabled else hard_labels)
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()
            running += float(loss.detach()) * self.grad_accum

            if (i + 1) % self.grad_accum == 0 or (i + 1) == batches_per_epoch:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                step += 1

            tracker.update(logits, hard_labels, loss=running / (i + 1))
            if (i + 1) % self.log_every == 0:
                pbar.set_postfix(loss=f"{running / (i + 1):.4f}")

            if self.requeue.should_stop:
                log.warning("Stop flag set during training epoch %d, step %d.", epoch, i)
                self.save_resumable(epoch - 1)  # safest: mark as not-yet-finished
                break

        summary = tracker.summary()
        summary["train_loss"] = running / max(1, (i + 1))
        return summary

    @torch.no_grad()
    def _eval_epoch(self, epoch: int, loader: DataLoader, tag: str = "val") -> dict:
        self.model.eval()
        tracker = MetricTracker()
        total_batches = len(loader)
        if self._debug and tag == "val" and self._max_val_batches:
            total_batches = min(total_batches, int(self._max_val_batches))

        pbar = tqdm(loader, desc=f"{tag}[{epoch}]", total=total_batches, leave=False)
        for i, batch in enumerate(pbar):
            if self._debug and tag == "val" and self._max_val_batches and i >= int(self._max_val_batches):
                break
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.amp):
                logits = self.model(pixel_values)
                loss = self.loss_fn(logits, labels)
            tracker.update(logits, labels, loss=float(loss.detach()))
        return tracker.summary()

    # ------------------------------------------------------------------
    # Top-level fit
    # ------------------------------------------------------------------
    def fit(self) -> EpochHistory:
        self.model.to(self.device)
        # Move loss buffers (class weights) to device.
        self.loss_fn.to(self.device)

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_stats = self._train_epoch(epoch)
            val_stats = self._eval_epoch(epoch, self.val_loader, tag="val")

            if self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            log.info(
                "epoch %3d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f f1 %.4f | lr %.2e | %.1fs",
                epoch, self.epochs,
                train_stats["train_loss"], train_stats["accuracy"],
                val_stats["loss"], val_stats["accuracy"], val_stats["f1_macro"],
                lr, time.time() - t0,
            )

            # Track history.
            self.history.train_loss.append(train_stats["train_loss"])
            self.history.train_acc.append(train_stats["accuracy"])
            self.history.val_loss.append(val_stats["loss"])
            self.history.val_acc.append(val_stats["accuracy"])
            self.history.val_f1.append(val_stats["f1_macro"])
            self.history.lr.append(lr)

            if self.tb:
                self.tb.add_scalar("train/loss", train_stats["train_loss"], epoch)
                self.tb.add_scalar("train/accuracy", train_stats["accuracy"], epoch)
                self.tb.add_scalar("val/loss", val_stats["loss"], epoch)
                self.tb.add_scalar("val/accuracy", val_stats["accuracy"], epoch)
                self.tb.add_scalar("val/f1_macro", val_stats["f1_macro"], epoch)
                self.tb.add_scalar("optim/lr", lr, epoch)

            # Best-model tracking on the chosen monitor metric.
            monitor_val = {
                "val_accuracy": val_stats["accuracy"],
                "val_loss": -val_stats["loss"],
                "val_f1_macro": val_stats["f1_macro"],
            }.get(self.monitor, val_stats["accuracy"])
            if monitor_val > self.best_metric:
                self.best_metric = monitor_val
                self.save_best(epoch)
                log.info("  ↳ new best %s=%.4f — saved best.pth", self.monitor, monitor_val)

            # Periodic resumable.
            if (epoch + 1) % self.save_every == 0:
                self.save_resumable(epoch)

            # Early stopping.
            if self.early is not None:
                self.early.step(monitor_val)
                if self.early.should_stop:
                    log.info("Early stopping triggered at epoch %d.", epoch)
                    self.save_resumable(epoch)
                    break

            # SLURM requeue path.
            if self.requeue.should_stop:
                self.save_resumable(epoch)
                self.requeue.maybe_requeue()
                break

        return self.history
