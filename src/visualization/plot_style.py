"""Paper-ready matplotlib style.

Calling ``set_paper_style()`` once at the top of a visualization script gives
every subsequent figure consistent fonts, DPI, and colour choices, so the
figures in the paper look like they came from the same toolkit.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_paper_style(font_size: int = 11) -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.6,
    })


def savefig_dual(fig: "plt.Figure", path_without_ext) -> None:
    """Save a figure as both .pdf (for LaTeX) and .png (for preview)."""
    from pathlib import Path
    p = Path(path_without_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"))
    fig.savefig(p.with_suffix(".png"))
