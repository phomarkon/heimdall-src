"""Aggregated SHAP attribution heatmap across F8's 5 seeds.

Per docs/RESEARCH-PROPOSAL.md §1.3 + §5 — XAI is a primary deliverable. This figure
shows mean attribution per (input timestep, feature) averaged across the 5
frozen seeds, with a per-seed CI band overlaid as a horizontal strip plot of
the per-feature integrated attribution magnitude.

Saves ``figures/explain_forecast_aggregated.png``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from heimdall_forecaster.train.dataset import F8_FEATURES, SEQ_LEN, make_windows
from heimdall_forecaster.train.model import PatchTransformerQuantile
from heimdall_ml.explain.explain_forecast import explain
from heimdall_ml.viz import apply_paper_style, paper_palette
from heimdall_ml.viz.style import PAGE_WIDTH_IN

REPO_ROOT = Path(__file__).resolve().parents[3]
F8_ROOT = REPO_ROOT / "models/forecaster/f8"
FIG_DIR = REPO_ROOT / "figures"
FEATURE_NAMES = F8_FEATURES


def _load_one_seed(seed_dir: Path) -> tuple[PatchTransformerQuantile, object]:
    cfg = json.loads((seed_dir / "config.json").read_text())
    model = PatchTransformerQuantile(
        n_features=int(cfg["n_features"]),
        seq_len=int(cfg["seq_len"]),
        horizon=int(cfg["horizon"]),
        n_quantiles=3,
        patch_len=int(cfg["patch_len"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        n_layers=int(cfg["n_layers"]),
        dropout=0.0,
    )
    state = torch.load(seed_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(seed_dir / "stats.pkl", "rb") as fh:
        stats = pickle.load(fh)
    return model, stats


def _val_window(stats) -> np.ndarray:
    df = pl.read_parquet(REPO_ROOT / "data/processed/dk1_panel_val.parquet").drop_nulls()
    arr = df.select(FEATURE_NAMES).to_numpy().astype(np.float64)
    arr_norm = stats.normalise(arr)
    # take a representative mid-val window
    mid = arr_norm.shape[0] // 2
    return arr_norm[mid - SEQ_LEN : mid].astype(np.float32)


def main() -> int:
    apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    seed_dirs = sorted(F8_ROOT.glob("seed-*"))
    if not seed_dirs:
        print("no F8 seeds yet")
        return 1

    per_seed_values: list[np.ndarray] = []
    for sd in seed_dirs:
        model, stats = _load_one_seed(sd)
        x = _val_window(stats)
        ex = explain(
            model,
            x,
            feature_names=FEATURE_NAMES,
            target_horizon_step=0,
            quantile_idx=1,
        )
        per_seed_values.append(ex.values)

    stacked = np.stack(per_seed_values, axis=0)  # (S, T, F)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)

    fig = plt.figure(figsize=(PAGE_WIDTH_IN, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.4], wspace=0.35)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])

    abs_mean = np.abs(mean)
    vmax = float(abs_mean.max() + 1e-9)
    im = ax_h.imshow(
        mean.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        origin="lower",
    )
    ax_h.set_yticks(range(len(FEATURE_NAMES)))
    ax_h.set_yticklabels([f.replace("_", " ") for f in FEATURE_NAMES], fontsize=7)
    ax_h.set_xlabel("Input step (oldest → newest, 15-min)")
    ax_h.set_title(f"F8 SHAP attribution, mean over {stacked.shape[0]} seeds")
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.04, pad=0.02)
    cbar.set_label("Attribution to q50")

    # Per-feature integrated magnitude with CI strip plot.
    integ = np.abs(mean).sum(axis=0)
    seed_integ = np.abs(stacked).sum(axis=1)  # (S, F)
    palette = paper_palette(len(FEATURE_NAMES))
    for fi, name in enumerate(FEATURE_NAMES):
        ax_s.scatter(
            seed_integ[:, fi], np.full(seed_integ.shape[0], fi),
            color=palette[fi], alpha=0.6, s=15,
        )
        ax_s.scatter([integ[fi]], [fi], color="k", marker="x", s=40, zorder=5)
    ax_s.set_yticks(range(len(FEATURE_NAMES)))
    ax_s.set_yticklabels([f.replace("_", " ") for f in FEATURE_NAMES], fontsize=7)
    ax_s.set_xlabel("Σ|attribution|")
    ax_s.set_title("Per-feature CI", fontsize=8)

    fig.savefig(FIG_DIR / "explain_forecast_aggregated.png")
    plt.close(fig)
    print(f"-> {FIG_DIR / 'explain_forecast_aggregated.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
