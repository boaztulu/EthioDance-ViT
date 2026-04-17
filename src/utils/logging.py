"""Rich + stdlib logging with per-run log file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    from rich.logging import RichHandler
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger()
    # Remove duplicate handlers across repeated setup calls (e.g. notebooks).
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    if _HAS_RICH:
        console = RichHandler(rich_tracebacks=True, show_path=False, show_time=False)
        console.setFormatter(logging.Formatter("%(message)s"))
    else:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        file_h.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        root.addHandler(file_h)

    root.setLevel(level)
    # Silence noisy libraries.
    for noisy in ("PIL", "matplotlib", "urllib3", "h5py", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
