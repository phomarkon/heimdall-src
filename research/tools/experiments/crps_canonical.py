"""CRPS for the canonical price zoo.

CRPS approximation from the saved q10/q50/q90 quantiles via the discrete
quantile-decomposition (Laio & Tamea 2007): CRPS = E_F|X - y| - 0.5 E_F|X - X'|,
estimated by the trapezoidal integral on the 3 quantile levels. This under-
estimates the full CRPS but is consistent across models (same quantile grid)
so the relative ordering is preserved.

Writes outputs/crps/leaderboard.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/crps"
QUANTILES = (0.1, 0.5, 0.9)
SEEDS = (13, 42, 137, 1729, 31415)


def _crps_quantile(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Approximate CRPS via the quantile pinball average × 2 (Laio & Tamea 2007).

    CRPS(F, y) = 2 ∫_0^1 ρ_α(y - F^{-1}(α)) dα ≈ 2 · mean over the discrete
    quantile grid of pinball losses. Returns per-row CRPS (shape N).
    """
    crps_per_row = np.zeros(targets.shape[0], dtype=np.float64)
    for qi, q in enumerate(QUANTILES):
        err = targets - preds[..., qi]
        pin = np.maximum(q * err, (q - 1.0) * err)
        crps_per_row += pin.mean(axis=1) * 2.0 / len(QUANTILES)
    return crps_per_row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    models = ["canonical_f1_lgbm_price", "canonical_f7_price", "canonical_f8_price",
              "canonical_f11_price", "canonical_f0_price",
              "f5_canonical_f5_np_price", "f6_canonical_f6_anp_price"]
    rows = []
    for m in models:
        md = REPO / "models/forecaster" / m
        if not md.exists():
            continue
        per_seed = []
        for seed in SEEDS:
            npz = md / f"seed-{seed}" / "val_preds.npz"
            if not npz.exists():
                continue
            d = np.load(npz)
            if "preds" not in d or "targets" not in d:
                continue
            preds = d["preds"]; targets = d["targets"]
            if preds.ndim != 3 or preds.shape[-1] != 3:
                continue
            per_seed.append(float(_crps_quantile(preds, targets).mean()))
        if per_seed:
            rows.append({
                "model": m, "n_seeds": len(per_seed),
                "crps_mean": float(np.mean(per_seed)),
                "crps_std": float(np.std(per_seed)),
            })
            print(f"  {m}: CRPS {rows[-1]['crps_mean']:.2f} ± {rows[-1]['crps_std']:.2f}")
    rows.sort(key=lambda r: r["crps_mean"])
    (OUT / "leaderboard.json").write_text(json.dumps(rows, indent=2))
    print(f"[crps] wrote {OUT}/leaderboard.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
