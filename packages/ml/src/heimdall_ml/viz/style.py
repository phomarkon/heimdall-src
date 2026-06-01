"""NeurIPS-friendly matplotlib style. One-call setup for every figure script.

Per docs/RESEARCH-PROPOSAL.md §10 (reproducibility): all paper figures share the
same fonts, palette, and column-width assumptions so the LaTeX submission is
typographically clean.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt

# NeurIPS one-column width is ~3.25 in, two-column ~6.75 in.
COLUMN_WIDTH_IN = 3.25
PAGE_WIDTH_IN = 6.75


def apply_paper_style() -> None:
    """Apply once at the top of every figure script."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.0,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "image.cmap": "viridis",
        }
    )


def paper_palette(n: int = 6) -> list[str]:
    """Colour-blind-safe categorical palette for line plots; cycles for n > 9."""
    palette = [
        "#0072B2",  # blue
        "#D55E00",  # orange
        "#009E73",  # green
        "#CC79A7",  # pink
        "#F0E442",  # yellow
        "#56B4E9",  # sky
        "#E69F00",  # tan
        "#000000",  # black
        "#882255",  # plum
    ]
    if n <= len(palette):
        return palette[: max(n, 1)]
    out = []
    for i in range(n):
        out.append(palette[i % len(palette)])
    return out


def horizon_axis(ax: plt.Axes, n_steps: int) -> None:
    """Convention used across the paper: horizon ticks every 4 steps (1 h)."""
    ax.set_xlabel("Forecast horizon (15-min steps)")
    ax.set_xticks(list(range(0, n_steps + 1, 4)))


__all__ = ["COLUMN_WIDTH_IN", "PAGE_WIDTH_IN", "apply_paper_style", "horizon_axis", "paper_palette"]
