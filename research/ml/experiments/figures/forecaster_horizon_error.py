"""Figure: mean pinball loss vs forecast-horizon step, per model.

Per docs/RESEARCH-PROPOSAL.md §5 (paper figures). One line per model in the F-zoo
leaderboard; loss is averaged across the three quantile levels at each horizon
step. Saved to ``figures/forecaster_horizon_error.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from heimdall_ml.viz import apply_paper_style, paper_palette
from heimdall_ml.viz.style import COLUMN_WIDTH_IN, horizon_axis

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = REPO_ROOT / "models/forecaster"
FIG_DIR = REPO_ROOT / "figures"
QUANTILES = (0.1, 0.5, 0.9)


def _per_horizon_pinball(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return shape (H,) of mean pinball across quantiles, averaged over windows."""
    losses = np.empty(preds.shape[1], dtype=np.float64)
    for h in range(preds.shape[1]):
        lh = []
        for qi, q in enumerate(QUANTILES):
            err = targets[:, h] - preds[:, h, qi]
            lh.append(np.mean(np.maximum(q * err, (q - 1.0) * err)))
        losses[h] = np.mean(lh)
    return losses


def _read_first_seed(model_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    for seed_dir in sorted((MODEL_ROOT / model_name).glob("seed-*")):
        f = seed_dir / "val_preds.npz"
        if f.exists():
            z = np.load(f)
            return z["preds"], z["targets"]
    return None


def main() -> int:
    apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    candidates = ["b1_random_walk", "b2_ewma", "b3_seasonal_naive", "b4_lightgbm_quantile",
                  "b7_nbeats_lite", "f0", "f3", "f7", "f8"]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
    palette = paper_palette(len(candidates))

    plotted = 0
    for name, color in zip(candidates, palette, strict=False):
        out = _read_first_seed(name)
        if out is None:
            continue
        preds, targets = out
        loss = _per_horizon_pinball(preds, targets)
        ax.plot(np.arange(1, loss.size + 1), loss, label=name.upper().replace("_", " "), color=color)
        plotted += 1

    horizon_axis(ax, 16)
    ax.set_ylabel("Mean pinball (DKK/MWh)")
    ax.set_title(f"F-zoo: pinball vs horizon (n={plotted} models)")
    ax.legend(ncol=2, loc="upper left", fontsize=6.5)
    out_path = FIG_DIR / "forecaster_horizon_error.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
