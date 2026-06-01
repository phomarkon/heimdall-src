"""A8b — BOCPD-augmented ACI vs vanilla ACI across the March 4, 2025 break.

The Theorem 1c depth play (per the 2026-05-10 strategy session): vanilla
online ACI converges to nominal coverage at rate O(1/(γT)) (Gibbs–Candès
Prop 4.1).  Pairing ACI with explicit Bayesian online change-point
detection (BOCPD; Adams & MacKay 2007) lets us *reset* the calibration
buffer on detected breaks, so post-break coverage recovers within ~96
ticks of detection, not ~1/γ × T.

This script:
  1. assembles a panel that spans the natural-experiment break
     (last 30 days of train + the full val window, ≈ 8 448 ticks);
  2. fits a naive AR(24) point forecaster (cheap, no training);
  3. runs three calibrators on the same point forecasts:
       a. vanilla split-CP with a fixed pre-break calibration buffer
          (the "deployed model" worst case);
       b. vanilla online ACI (Gibbs–Candès);
       c. BOCPD-ACI: ACI with a reset-on-detection trigger;
  4. computes rolling 96-tick coverage and writes
     `experiments/outputs/a8b_bocpd_aci.json` + a CSV figure-source.

The output is the empirical anchor for Theorem 1c.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from heimdall_ml.conformal.bocpd import BOCPD

REPO_ROOT = Path(__file__).resolve().parents[3]
ALPHA = 0.10
ROLLING_WINDOW = 96
GAMMA = 0.05
CAL_BUFFER_PRE = 96 * 30  # 30 days of train tail
ACI_MIN_BUFFER = 96 * 2   # never shrink below 2 days post-reset


# ---------------------------------------------------------------------------
# Naive AR(24) point forecaster (matches F0 in the zoo).
# ---------------------------------------------------------------------------


def ar24_point(history: np.ndarray) -> float:
    """Predict the next 15-min imbalance price as a 24-h-ago seasonal lag."""
    if history.size >= 96:
        return float(history[-96])
    if history.size:
        return float(history[-1])
    return 0.0


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------


@dataclass
class SplitCP:
    """Fixed split-CP calibration on a frozen buffer of nonconformity scores."""
    scores: np.ndarray
    alpha: float = ALPHA

    def interval(self, point: float) -> tuple[float, float]:
        if self.scores.size == 0:
            return (point - 100.0, point + 100.0)
        q = float(np.quantile(self.scores, 1.0 - self.alpha))
        return (point - q, point + q)


@dataclass
class OnlineACI:
    """Vanilla Gibbs-Candès ACI."""
    alpha_target: float = ALPHA
    gamma: float = GAMMA
    alpha_t: float = ALPHA
    scores: list = None

    def __post_init__(self):
        if self.scores is None:
            self.scores = []

    def interval(self, point: float) -> tuple[float, float]:
        if not self.scores:
            return (point - 100.0, point + 100.0)
        a = float(np.clip(self.alpha_t, 1e-3, 1.0 - 1e-3))
        q = float(np.quantile(self.scores, 1.0 - a))
        return (point - q, point + q)

    def update(self, realised: float, point: float) -> None:
        s = abs(realised - point)
        self.scores.append(s)
        in_band = s <= float(np.quantile(self.scores, 1.0 - max(self.alpha_t, 1e-3)))
        err = 0.0 if in_band else 1.0
        self.alpha_t = float(np.clip(
            self.alpha_t + self.gamma * (self.alpha_target - err), 1e-3, 1 - 1e-3
        ))


@dataclass
class BOCPDOnlineACI(OnlineACI):
    """ACI augmented by BOCPD: on detected change-point, reset the buffer."""
    bocpd: BOCPD = None
    min_buffer: int = ACI_MIN_BUFFER
    last_reset: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.bocpd is None:
            self.bocpd = BOCPD(mean_run_length=200.0)

    def update(self, realised: float, point: float, t: int = 0) -> bool:
        super().update(realised, point)
        # Feed the *residual* to BOCPD (the score series).
        r = self.bocpd.step(float(realised - point))
        if r.detected_change and (t - self.last_reset) > self.min_buffer:
            # Reset: restrict scores to the most recent `min_buffer` and reset α.
            self.scores = list(self.scores[-self.min_buffer:])
            self.alpha_t = self.alpha_target
            self.last_reset = t
            return True
        return False


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run() -> dict:
    train = pl.read_parquet(REPO_ROOT / "data/processed/dk1_panel_train.parquet").sort("timestamp_utc")
    val = pl.read_parquet(REPO_ROOT / "data/processed/dk1_panel_val.parquet").sort("timestamp_utc")
    panel = pl.concat([train.tail(96 * 30), val], how="vertical")
    imb = panel["imbalance_price_dkk_mwh_15min"].to_numpy().astype(np.float64)
    ts = panel["timestamp_utc"].to_list()
    T = imb.size

    # Pre-warm: build calibration buffer from the first 30 days (pre-break).
    warmup_n = 96 * 30
    warmup_scores = []
    for t in range(96, warmup_n):
        warmup_scores.append(abs(imb[t] - ar24_point(imb[:t])))
    warmup_scores = np.array(warmup_scores)

    split_cp = SplitCP(scores=warmup_scores.copy(), alpha=ALPHA)
    aci = OnlineACI(scores=list(warmup_scores))
    bocpd_aci = BOCPDOnlineACI(scores=list(warmup_scores))

    in_band = {"split_cp": [], "aci": [], "bocpd_aci": []}
    timestamps_eval = []
    bocpd_resets = []

    for t in range(warmup_n, T):
        history = imb[:t]
        point = ar24_point(history)
        realised = imb[t]

        for name, cal in [("split_cp", split_cp), ("aci", aci), ("bocpd_aci", bocpd_aci)]:
            lo, hi = cal.interval(point)
            in_band[name].append(int(lo <= realised <= hi))

        # split_cp is *frozen* (no update) — that's the "deployed model" point.
        aci.update(realised=realised, point=point)
        if bocpd_aci.update(realised=realised, point=point, t=t):
            bocpd_resets.append({"t": t, "ts": ts[t].isoformat() if hasattr(ts[t], "isoformat") else str(ts[t])})
        timestamps_eval.append(str(ts[t]))

    # Rolling coverage (96-tick).
    def rolling(arr: list[int], w: int = ROLLING_WINDOW) -> list[float]:
        a = np.array(arr, dtype=float)
        cs = np.concatenate([[0.0], np.cumsum(a)])
        out = np.empty_like(a)
        for i in range(a.size):
            lo = max(0, i - w + 1)
            out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
        return out.tolist()

    rolling_cov = {k: rolling(v) for k, v in in_band.items()}

    # Mark the break-point: the val window starts at 2025-03-04 00:00 UTC.
    # In the warmup-then-eval frame, the break is at index (warmup_n - 30*96).
    # warmup_n = 30*96 = 2880; train.tail(30*96) ends right before val starts,
    # so the break index in the *eval* frame is t=0 (we begin evaluating
    # immediately at the break).
    break_idx = 0

    summary = {
        "alpha_target": ALPHA,
        "n_eval_ticks": T - warmup_n,
        "warmup_ticks": warmup_n,
        "rolling_window": ROLLING_WINDOW,
        "gamma": GAMMA,
        "n_bocpd_resets": len(bocpd_resets),
        "bocpd_first_reset_t": bocpd_resets[0]["t"] if bocpd_resets else None,
        "marginal_cov": {k: float(np.mean(v)) for k, v in in_band.items()},
        "post_break_first_24h_cov": {
            k: float(np.mean(v[:96])) for k, v in in_band.items()
        },
        "post_break_first_72h_cov": {
            k: float(np.mean(v[:96 * 3])) for k, v in in_band.items()
        },
    }
    print(json.dumps(summary, indent=2))

    out = {
        "summary": summary,
        "rolling_cov": rolling_cov,
        "bocpd_resets": bocpd_resets,
        "timestamps_eval": timestamps_eval,
    }
    out_dir = REPO_ROOT / "experiments/outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a8b_bocpd_aci.json").write_text(json.dumps(out, indent=2))

    # CSV companion for figure regeneration.
    import csv
    with open(out_dir / "a8b_bocpd_aci_rolling.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "timestamp", "split_cp", "aci", "bocpd_aci"])
        for i in range(len(rolling_cov["aci"])):
            w.writerow([
                i,
                timestamps_eval[i],
                rolling_cov["split_cp"][i],
                rolling_cov["aci"][i],
                rolling_cov["bocpd_aci"][i],
            ])
    return out


if __name__ == "__main__":
    run()
