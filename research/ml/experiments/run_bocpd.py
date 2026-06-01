"""Run BOCPD on F8b seed-13 validation-window residuals.

Computes residual_t = y_t - q50_t (first-horizon, flattened) and runs
the existing heimdall_ml.conformal.bocpd.BOCPD detector. Persists the
full run-length posterior (T × R) and the detected change-point indices
to experiments/outputs/bocpd_runlength.npz for the appendix heatmap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from heimdall_ml.conformal.bocpd import BOCPD

REPO = Path(__file__).resolve().parents[2]
SEED = 13
SRC = REPO / "models" / "forecaster" / "f8b" / f"seed-{SEED}" / "val_preds.npz"
OUT = REPO / "experiments" / "outputs" / "bocpd_runlength.npz"


def main() -> int:
    d = np.load(SRC)
    preds = d["preds"]; targets = d["targets"]
    # First-horizon residual gives one observation per issue-time window.
    resid = (targets[:, 0] - preds[:, 0, 1]).astype(np.float64)
    # Standardise for numerical stability of NIG predictive.
    z = (resid - resid.mean()) / (resid.std() + 1e-9)

    bocpd = BOCPD(mean_run_length=200.0,
                  detection_threshold=4, detection_prev_threshold=32)
    T = z.size
    R_MAX = 500  # cap for visualization; longer runs collapse into the top row
    posterior = np.zeros((T, R_MAX), dtype=np.float32)
    detected: list[int] = []
    for t, x in enumerate(z):
        r = bocpd.step(float(x))
        p = r.posterior
        k = min(p.size, R_MAX)
        posterior[t, :k] = p[:k]
        if r.detected_change:
            detected.append(t)

    # Marginal P(r_t = 0) at each step.
    p_r0 = posterior[:, 0]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, posterior=posterior, p_r0=p_r0,
             detected=np.asarray(detected, dtype=np.int64),
             resid=resid.astype(np.float32))
    print(f"wrote {OUT}  T={T}  detected={len(detected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
