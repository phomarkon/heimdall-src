"""ACI γ-sweep over the F8b validation window.

For each γ ∈ {0.001, 0.005, 0.01, 0.05} we run the Gibbs–Candès online
ACI update γ_{t+1}=γ_t+η(α-1{y∉C}) on top of the raw f8b quantile head
(seed 13) and persist the rolling-coverage trajectory.

Output: experiments/outputs/aci_gamma_sweep.json
   {gamma: [{step: int, coverage: float}, ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEED = 13
ALPHA = 0.1
ROLL = 96
GAMMAS = (0.001, 0.005, 0.01, 0.05)


def aci_hits(preds: np.ndarray, targets: np.ndarray,
             alpha: float, gamma: float) -> np.ndarray:
    """Per-step horizon-mean hit rate under online ACI."""
    q10 = preds[..., 0]; q50 = preds[..., 1]; q90 = preds[..., 2]
    half_up = np.maximum(q90 - q50, 1e-6)
    half_dn = np.maximum(q50 - q10, 1e-6)
    resid = targets - q50
    norm = np.where(resid >= 0, resid / half_up, resid / half_dn)
    abs_norm = np.abs(norm)
    T = abs_norm.shape[0]
    s = float(np.quantile(abs_norm, 1.0 - alpha))
    hits = np.zeros(T, dtype=np.float32)
    for t in range(T):
        inside = abs_norm[t] <= s
        hits[t] = inside.mean()
        err = 1.0 - hits[t]
        s = max(s + gamma * (alpha - err), 1e-6)
    return hits


def rolling(x: np.ndarray, win: int) -> np.ndarray:
    cs = np.cumsum(np.insert(x, 0, 0.0))
    return (cs[win:] - cs[:-win]) / win


def main() -> int:
    src = REPO / "models" / "forecaster" / "f8b" / f"seed-{SEED}" / "val_preds.npz"
    d = np.load(src)
    out: dict[str, list] = {}
    for g in GAMMAS:
        hits = aci_hits(d["preds"], d["targets"], alpha=ALPHA, gamma=g)
        cov = rolling(hits, ROLL)
        out[f"{g:.4f}"] = [
            {"step": int(i), "coverage": float(c)} for i, c in enumerate(cov)
        ]
        print(f"γ={g:.4f}  mean cov={cov.mean():.3f}  n={len(cov)}")

    target = REPO / "experiments" / "outputs" / "aci_gamma_sweep.json"
    target.write_text(json.dumps(out))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
