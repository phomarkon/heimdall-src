"""Diebold-Mariano test for pinball-loss differences between price forecasters.

Uses saved val_preds.npz per (model, seed) and computes the DM statistic
under the standard heteroscedasticity-and-autocorrelation-consistent (HAC)
variance estimator. Reports two-sided p-values for all pairs against F1 LGBM
(the test-set winner) on the validation window.

Writes outputs/dm_test/results.json.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs/dm_test"
QUANTILES = (0.1, 0.5, 0.9)
SEEDS = (13, 42, 137, 1729, 31415)


def _pinball_per_row(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return mean pinball over (H, Q) per window — shape (N,)."""
    per_q = np.zeros_like(targets, dtype=np.float64)
    out = np.zeros(targets.shape[0], dtype=np.float64)
    for qi, q in enumerate(QUANTILES):
        err = targets - preds[..., qi]
        per_q = np.maximum(q * err, (q - 1.0) * err)
        out += per_q.mean(axis=1)
    return out / len(QUANTILES)


def _hac_variance(d: np.ndarray, lag: int) -> float:
    """Newey-West HAC long-run variance with Bartlett kernel, lag=lag."""
    n = len(d); dm = d - d.mean()
    gamma0 = float((dm @ dm) / n)
    s = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        gk = float((dm[k:] @ dm[:-k]) / n)
        s += 2 * w * gk
    return max(s, 1e-12)


def _dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lag: int | None = None) -> dict:
    d = loss_a - loss_b
    n = len(d)
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    sigma2 = _hac_variance(d, lag)
    dm = float(d.mean() / np.sqrt(sigma2 / n))
    # two-sided p-value via standard normal
    from math import erf
    p = float(2 * (1 - 0.5 * (1 + erf(abs(dm) / np.sqrt(2)))))
    return {"dm_stat": dm, "p_value": p, "lag": lag,
            "mean_loss_a": float(loss_a.mean()),
            "mean_loss_b": float(loss_b.mean()),
            "mean_diff": float(d.mean())}


def _load_losses(model_dir: Path) -> np.ndarray | None:
    losses = []
    for seed in SEEDS:
        npz = model_dir / f"seed-{seed}" / "val_preds.npz"
        if not npz.exists():
            return None
        d = np.load(npz)
        if "preds" not in d or "targets" not in d:
            return None
        preds = d["preds"]; targets = d["targets"]
        if preds.ndim != 3 or preds.shape[-1] != 3:
            return None
        losses.append(_pinball_per_row(preds, targets))
    # pad to common length, then average across seeds per row
    L = min(len(l) for l in losses)
    arr = np.stack([l[:L] for l in losses])
    return arr.mean(axis=0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    models = ["canonical_f1_lgbm_price", "canonical_f7_price", "canonical_f8_price",
              "canonical_f11_price", "canonical_f0_price",
              "f5_canonical_f5_np_price", "f6_canonical_f6_anp_price"]
    losses: dict[str, np.ndarray] = {}
    for m in models:
        md = REPO / "models/forecaster" / m
        if not md.exists():
            continue
        l = _load_losses(md)
        if l is not None:
            losses[m] = l
            print(f"  {m}: mean pinball {l.mean():.2f} (n={len(l)})", flush=True)
    # Pair every model against F1 (the winner)
    anchor = "canonical_f1_lgbm_price"
    results = {}
    if anchor in losses:
        for m, l in losses.items():
            if m == anchor: continue
            la = losses[anchor]; lb = l
            n = min(len(la), len(lb))
            r = _dm_test(la[:n], lb[:n])
            r["model_a"] = anchor; r["model_b"] = m
            results[f"{anchor}_vs_{m}"] = r
            print(f"  DM {anchor} vs {m}: dm={r['dm_stat']:.2f} p={r['p_value']:.4f}", flush=True)
    # All pairs too
    all_pairs = {}
    for a, b in combinations(losses.keys(), 2):
        la, lb = losses[a], losses[b]
        n = min(len(la), len(lb))
        r = _dm_test(la[:n], lb[:n])
        all_pairs[f"{a}_vs_{b}"] = r

    (OUT / "results.json").write_text(json.dumps({"vs_anchor": results, "all_pairs": all_pairs}, indent=2))
    print(f"[dm] wrote {OUT}/results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
