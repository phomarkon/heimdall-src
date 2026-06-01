"""A8c — BOCPD-augmented ACI on the *deployed* F7 forecaster.

Strengthens the empirical anchor for Theorem 1c: the day-1 A8b run used
the F0 naive-AR(24) point forecaster (cheap, no training, but not the
deployed model).  This ablation replaces F0 with the trained
patch-transformer F7 (5-seed mean q50 from
``models/forecaster/f7/seed-*/val_preds.npz``) and re-runs the same
three calibrators (frozen split-CP, vanilla ACI, BOCPD-ACI) over the
post-break val window.

Output:
- ``experiments/outputs/a8c_bocpd_on_f7_aci.json``
- ``experiments/outputs/a8c_bocpd_on_f7_aci_rolling.csv``

Compared to A8b's F0 anchor: F7's residuals are smaller and more
stationary within the post-break regime, so the *gap* between vanilla
ACI and BOCPD-ACI should shrink (less to recover).  Reports a clean
ablation either way.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from heimdall_ml.conformal.bocpd import BOCPD

REPO_ROOT = Path(__file__).resolve().parents[3]
ALPHA = 0.10
GAMMA = 0.05
ROLLING_WINDOW = 96
ACI_MIN_BUFFER = 96 * 2
DEFAULT_FORECASTER_DIR = REPO_ROOT / "models/forecaster/f7"
SEEDS = (13, 42, 137, 1729, 31415)


def _load_f7_predictions(horizon_step: int = 0, forecaster_dir: Path = DEFAULT_FORECASTER_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Return (q50_mean, targets) over the val window — averaged across seeds.

    ``horizon_step`` selects the prediction step within the 16-step
    forecast horizon; 0 = next 15-min tick.
    """
    preds_per_seed = []
    targets = None
    for s in SEEDS:
        npz = np.load(forecaster_dir / f"seed-{s}" / "val_preds.npz")
        preds_per_seed.append(npz["preds"][:, horizon_step, 1].astype(np.float64))  # q50
        if targets is None:
            targets = npz["targets"][:, horizon_step].astype(np.float64)
    q50 = np.mean(np.stack(preds_per_seed, axis=0), axis=0)
    return q50, targets


def _rolling(arr: np.ndarray, w: int = ROLLING_WINDOW) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(a)])
    out = np.empty_like(a)
    for i in range(a.size):
        lo = max(0, i - w + 1)
        out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
    return out


def run(forecaster_name: str = "f7") -> dict:
    forecaster_dir = REPO_ROOT / "models/forecaster" / forecaster_name
    point, realised = _load_f7_predictions(horizon_step=0, forecaster_dir=forecaster_dir)
    n = point.size
    assert realised is not None and realised.size == n
    scores = np.abs(realised - point)

    # Pre-warm with first 30% of the series as a frozen calibration buffer.
    warmup_n = max(96 * 7, n // 5)
    pre = scores[:warmup_n].copy()
    eval_real = realised[warmup_n:]
    eval_point = point[warmup_n:]
    n_eval = eval_real.size

    # Frozen split-CP.
    q_split = float(np.quantile(pre, 1.0 - ALPHA))
    in_split = ((eval_point - q_split) <= eval_real) & (eval_real <= (eval_point + q_split))

    # Vanilla ACI.
    aci_scores = list(pre)
    alpha_t = ALPHA
    in_aci = np.zeros(n_eval, dtype=int)
    for i in range(n_eval):
        a = float(np.clip(alpha_t, 1e-3, 1 - 1e-3))
        q = float(np.quantile(aci_scores, 1.0 - a))
        in_aci[i] = int((eval_point[i] - q) <= eval_real[i] <= (eval_point[i] + q))
        aci_scores.append(abs(eval_real[i] - eval_point[i]))
        err = 0.0 if in_aci[i] == 1 else 1.0
        alpha_t = float(np.clip(alpha_t + GAMMA * (ALPHA - err), 1e-3, 1 - 1e-3))

    # BOCPD-ACI.
    bocpd_scores = list(pre)
    alpha_t_b = ALPHA
    in_bocpd = np.zeros(n_eval, dtype=int)
    bocpd = BOCPD(mean_run_length=200.0)
    last_reset = 0
    bocpd_resets: list[int] = []
    for i in range(n_eval):
        a = float(np.clip(alpha_t_b, 1e-3, 1 - 1e-3))
        q = float(np.quantile(bocpd_scores, 1.0 - a))
        in_bocpd[i] = int((eval_point[i] - q) <= eval_real[i] <= (eval_point[i] + q))
        residual = eval_real[i] - eval_point[i]
        bocpd_scores.append(abs(residual))
        r = bocpd.step(float(residual))
        err = 0.0 if in_bocpd[i] == 1 else 1.0
        alpha_t_b = float(np.clip(alpha_t_b + GAMMA * (ALPHA - err), 1e-3, 1 - 1e-3))
        if r.detected_change and (i - last_reset) > ACI_MIN_BUFFER:
            bocpd_scores = list(bocpd_scores[-ACI_MIN_BUFFER:])
            alpha_t_b = ALPHA
            last_reset = i
            bocpd_resets.append(i)

    summary = {
        "alpha_target": ALPHA,
        "n_eval_ticks": int(n_eval),
        "warmup_ticks": int(warmup_n),
        "rolling_window": ROLLING_WINDOW,
        "gamma": GAMMA,
        "n_seeds_averaged": len(SEEDS),
        "n_bocpd_resets": len(bocpd_resets),
        "first_bocpd_reset_t": bocpd_resets[0] if bocpd_resets else None,
        "marginal_cov": {
            "split_cp": float(np.mean(in_split)),
            "aci": float(np.mean(in_aci)),
            "bocpd_aci": float(np.mean(in_bocpd)),
        },
        "post_break_first_24h_cov": {
            "split_cp": float(np.mean(in_split[:96])),
            "aci": float(np.mean(in_aci[:96])),
            "bocpd_aci": float(np.mean(in_bocpd[:96])),
        },
        "post_break_first_72h_cov": {
            "split_cp": float(np.mean(in_split[: 96 * 3])),
            "aci": float(np.mean(in_aci[: 96 * 3])),
            "bocpd_aci": float(np.mean(in_bocpd[: 96 * 3])),
        },
    }
    print(json.dumps(summary, indent=2))

    rolling_cov = {
        "split_cp": _rolling(in_split.astype(float)).tolist(),
        "aci":      _rolling(in_aci.astype(float)).tolist(),
        "bocpd_aci": _rolling(in_bocpd.astype(float)).tolist(),
    }
    out = {"summary": summary, "rolling_cov": rolling_cov, "bocpd_resets": bocpd_resets}
    out_dir = REPO_ROOT / "experiments/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if forecaster_name == "f7" else f"_{forecaster_name}"
    (out_dir / f"a8c_bocpd_on_f7_aci{suffix}.json").write_text(json.dumps(out, indent=2))
    with open(out_dir / f"a8c_bocpd_on_f7_aci{suffix}_rolling.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "split_cp", "aci", "bocpd_aci"])
        for i in range(n_eval):
            w.writerow([i, rolling_cov["split_cp"][i], rolling_cov["aci"][i], rolling_cov["bocpd_aci"][i]])
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--forecaster", default="f7", choices=["f7", "f8"])
    args = p.parse_args()
    run(forecaster_name=args.forecaster)
