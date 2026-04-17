"""Vanilla early stopping keyed on any monitored metric."""

from __future__ import annotations

import math


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.0, mode: str = "max") -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved
