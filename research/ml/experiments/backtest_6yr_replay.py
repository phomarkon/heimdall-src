"""6-year DK1 replay back-test: Heimdall-guarded vs unguarded bidder.

No real money. No live trading. Pure replay-based simulation on real
Energinet DK1 historical data 2020-2026:

For every 15-minute quarter t in the 6-year history:
  1. A simple parametric bidder generates a candidate bid `a_t`
     (limit-order: side from EWMA-direction signal, qty = 10 MW fixed,
     limit price = EWMA forecast + safety margin).
  2. The Heimdall verifier evaluates `a_t` against the F3-ensemble +
     ACI-wrapped interval, accepting iff pi_min(a) >= tau.
  3. We compute the realised profit `pi(a_t, p_t)` at the real
     settle price p_t.

Aggregate:
  - mean realised profit per quarter (unguarded vs guarded)
  - cumulative profit time-series
  - 95% bootstrap CI on the mean gap
  - distribution of rejected-bid realised profits (what we'd have lost)

Headline question for the paper: across 220,000+ historical quarters,
does the Heimdall verifier improve realised profit vs the same bidder
without the verifier? At what cost (rejected throughput)?

Output:
  experiments/outputs/backtest_6yr_replay.json
  notes/findings/2026-05-17-backtest-6yr.md  (summary)
"""
from __future__ import annotations
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from heimdall_contracts import BidAction, ConformalInterval
from heimdall_markets import realized_profit, worst_case_profit
from heimdall_verifier.conformal import conformal_check

REPO = Path(__file__).resolve().parents[2]
FULL_PANEL = REPO / "data/processed/dk1_panel.parquet"  # 2020-01-01 -> 2026-04-29
F7_VAL_PREDS = REPO / "models/forecaster/f3_ensemble/seed-42/val_preds.npz"

# Bidder hyperparams: simple EWMA bidder. Not the LLM; the LLM is Tim's
# track and not required for this experiment.
EWMA_ALPHA = 0.10           # smoothing on imbalance price history
SEQ_LEN = 96                # 1-day rolling window
QTY_MW = 10.0               # fixed bid size
SAFETY_MARGIN_EUR = 50.0    # bidder shifts limit away from EWMA by this
TAU_EUR_SWEEP = (0.0, 10.0, 50.0, 100.0, 250.0)  # sweep verifier threshold
ALPHA = 0.10                # conformal target miscoverage
_REF_QUARTER = datetime(2025, 3, 4, 0, 0, tzinfo=timezone.utc)


def _ewma(series: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def _bid_for_quarter(history: np.ndarray, ewma_state: float,
                     rng: np.random.Generator, noise_scale_eur: float,
                     hallucination_rate: float = 0.0,
                     hallucination_scale_eur: float = 5000.0) -> tuple[BidAction, bool]:
    """EWMA-driven limit-order bidder with bid-level noise + LLM hallucinations.

    With probability `hallucination_rate` the bidder emits a HALLUCINATED bid:
    limit price drawn from a wide Gaussian centered on the EWMA, modelling
    the kind of catastrophic mistake an unguarded LLM agent can make. The
    verifier's role is to filter these.

    Returns (bid, is_hallucination).
    """
    is_hallu = bool(hallucination_rate > 0 and rng.uniform() < hallucination_rate)
    recent = float(history[-1])
    direction = "sell" if recent > ewma_state else "buy"
    if is_hallu:
        # Hallucinated bid: extreme limit price, wrong direction sometimes
        limit = ewma_state + float(rng.normal(0.0, hallucination_scale_eur))
        if rng.uniform() < 0.5:
            direction = "buy" if direction == "sell" else "sell"
    else:
        base = ewma_state + (SAFETY_MARGIN_EUR if direction == "sell" else -SAFETY_MARGIN_EUR)
        if noise_scale_eur > 0.0:
            limit = base + float(rng.normal(0.0, noise_scale_eur))
        else:
            limit = base
    return BidAction(
        market="mFRR", direction=direction, quantity_mw=QTY_MW,
        price_eur_per_mwh=float(limit),
        delivery_quarter=_REF_QUARTER,
    ), is_hallu


def _conformal_interval_for_quarter(
    history: np.ndarray, calibration_residuals: np.ndarray, ewma_state: float
) -> tuple[float, float]:
    """Closed-form split-CP interval around the EWMA forecast.

    Uses the SAME residual quantile q as the conformal calibrator would
    apply for alpha = 0.10 (90% interval). Calibration_residuals is a
    rolling buffer of recent |y - q50| residuals.
    """
    if len(calibration_residuals) < 50:
        # Cold start: wide interval to avoid spurious rejections.
        return ewma_state - 500.0, ewma_state + 500.0
    q = float(np.quantile(calibration_residuals, 1.0 - ALPHA))
    return ewma_state - q, ewma_state + q


NOISE_SCALE_EUR = 300.0       # bid-level noise scale (mimics LLM behavioural variability)
HALLUCINATION_RATE = 0.05     # fraction of bids that are LLM hallucinations
HALLUCINATION_SCALE_EUR = 5000.0  # hallucinated bids have extreme limit-price deviation


def main() -> int:
    print(f"[backtest 6yr] loading {FULL_PANEL} ...")
    df = pl.read_parquet(FULL_PANEL).sort("timestamp_utc").drop_nulls(["imbalance_price_dkk_mwh_15min"])
    prices = df["imbalance_price_dkk_mwh_15min"].to_numpy().astype(np.float64)
    timestamps = df["timestamp_utc"].to_numpy()
    n = len(prices)
    print(f"[backtest 6yr] N={n} quarters from {timestamps[0]} to {timestamps[-1]}")
    rng = np.random.default_rng(42)
    print(f"[backtest 6yr] noisy bidder with noise_scale={NOISE_SCALE_EUR} EUR")

    # Pre-compute EWMA over the full series (causal).
    ewma = _ewma(prices, EWMA_ALPHA)

    # Walk forward.
    cal_residuals = []  # rolling buffer of |y_t - ewma_t|
    cal_window = 2 * 7 * 96  # 2-week trailing window for the conformal cal

    n_eval = 0
    unguarded_profits: list[float] = []
    hallucinated_unguarded: list[float] = []  # unguarded profit on hallucinated bids only
    guarded_by_tau: dict[float, list[float]] = {t: [] for t in TAU_EUR_SWEEP}
    rejected_by_tau: dict[float, list[float]] = {t: [] for t in TAU_EUR_SWEEP}
    hallu_rejected_by_tau: dict[float, int] = {t: 0 for t in TAU_EUR_SWEEP}
    hallu_accepted_by_tau: dict[float, list[float]] = {t: [] for t in TAU_EUR_SWEEP}

    t0 = time.time()
    for i in range(SEQ_LEN, n - 1):
        hist = prices[i - SEQ_LEN: i]
        ewma_state = float(ewma[i - 1])
        cal_residuals.append(abs(prices[i - 1] - ewma_state))
        if len(cal_residuals) > cal_window:
            cal_residuals = cal_residuals[-cal_window:]

        bid, is_hallu = _bid_for_quarter(
            hist, ewma_state, rng, NOISE_SCALE_EUR,
            HALLUCINATION_RATE, HALLUCINATION_SCALE_EUR,
        )
        p_real = float(prices[i])
        unguarded = realized_profit(bid, p_real)
        unguarded_profits.append(unguarded)
        if is_hallu:
            hallucinated_unguarded.append(unguarded)

        lower, upper = _conformal_interval_for_quarter(hist, np.asarray(cal_residuals), ewma_state)
        if upper - lower < 1e-6:
            # Degenerate interval — append 0 to keep arrays aligned.
            for tau in TAU_EUR_SWEEP:
                guarded_by_tau[tau].append(0.0)
            continue
        interval = ConformalInterval(
            lower=lower, upper=upper, alpha=ALPHA, horizon_minutes=15, method="split_cp"
        )
        for tau in TAU_EUR_SWEEP:
            accepted, _ = conformal_check(bid, interval, tau_eur=tau)
            if accepted:
                guarded_by_tau[tau].append(unguarded)
                if is_hallu:
                    hallu_accepted_by_tau[tau].append(unguarded)
            else:
                guarded_by_tau[tau].append(0.0)
                rejected_by_tau[tau].append(unguarded)
                if is_hallu:
                    hallu_rejected_by_tau[tau] += 1
        n_eval += 1
        if n_eval % 20000 == 0:
            mid = TAU_EUR_SWEEP[len(TAU_EUR_SWEEP) // 2]
            print(f"  [{n_eval}/{n}] elapsed={time.time() - t0:.0f}s "
                  f"unguarded_mean={np.mean(unguarded_profits):.2f} "
                  f"guarded(@tau={mid})_mean={np.mean(guarded_by_tau[mid]):.2f}")

    runtime = time.time() - t0
    print(f"\n[backtest 6yr] done in {runtime:.0f}s, n_eval={n_eval}")

    rng = np.random.default_rng(42)
    n_boot = 1000
    unguarded_arr = np.asarray(unguarded_profits)
    out_rows = []
    for tau in TAU_EUR_SWEEP:
        guarded_arr = np.asarray(guarded_by_tau[tau])
        rej_arr = np.asarray(rejected_by_tau[tau])
        # Align in case of degenerate-skipped quarters.
        m = min(len(guarded_arr), len(unguarded_arr))
        paired = guarded_arr[:m] - unguarded_arr[:m]
        boot_means = np.array([paired[rng.integers(0, len(paired), len(paired))].mean()
                               for _ in range(n_boot)])
        row = {
            "tau_eur": tau,
            "n_evaluated": int(m),
            "unguarded_mean_eur": float(unguarded_arr[:m].mean()),
            "unguarded_sum_eur": float(unguarded_arr[:m].sum()),
            "unguarded_frac_negative": float((unguarded_arr[:m] < 0).mean()),
            "guarded_mean_eur": float(guarded_arr[:m].mean()),
            "guarded_sum_eur": float(guarded_arr[:m].sum()),
            "guarded_frac_accepted": float((guarded_arr[:m] != 0).mean()),
            "guarded_frac_negative": float((guarded_arr[:m] < 0).mean()),
            "gap_mean_eur": float(paired.mean()),
            "gap_bootstrap_ci_lo": float(np.quantile(boot_means, 0.025)),
            "gap_bootstrap_ci_hi": float(np.quantile(boot_means, 0.975)),
            "n_rejected": int(len(rej_arr)),
            "rejected_mean_if_accepted": float(rej_arr.mean()) if len(rej_arr) else 0.0,
            "rejected_frac_negative": float((rej_arr < 0).mean()) if len(rej_arr) else 0.0,
            "hallu_rejected_n": hallu_rejected_by_tau[tau],
            "hallu_accepted_n": len(hallu_accepted_by_tau[tau]),
            "hallu_filter_rate": (
                hallu_rejected_by_tau[tau] /
                max(1, hallu_rejected_by_tau[tau] + len(hallu_accepted_by_tau[tau]))
            ),
            "hallu_accepted_mean_profit": (
                float(np.mean(hallu_accepted_by_tau[tau])) if hallu_accepted_by_tau[tau] else 0.0
            ),
        }
        out_rows.append(row)
    # Also record unguarded hallucination stats
    hallu_un = np.asarray(hallucinated_unguarded) if hallucinated_unguarded else np.array([])

    out = {
        "config": {
            "panel": str(FULL_PANEL),
            "n_quarters_evaluated": n_eval,
            "ewma_alpha": EWMA_ALPHA,
            "seq_len": SEQ_LEN,
            "qty_mw": QTY_MW,
            "safety_margin_eur": SAFETY_MARGIN_EUR,
            "noise_scale_eur": NOISE_SCALE_EUR,
            "hallucination_rate": HALLUCINATION_RATE,
            "hallucination_scale_eur": HALLUCINATION_SCALE_EUR,
            "tau_sweep": list(TAU_EUR_SWEEP),
            "alpha": ALPHA,
            "calibration_window_quarters": cal_window,
        },
        "hallucinated_unguarded": {
            "n": int(len(hallu_un)),
            "mean": float(hallu_un.mean()) if len(hallu_un) else 0.0,
            "frac_negative": float((hallu_un < 0).mean()) if len(hallu_un) else 0.0,
            "sum_eur": float(hallu_un.sum()) if len(hallu_un) else 0.0,
            "q05": float(np.quantile(hallu_un, 0.05)) if len(hallu_un) else 0.0,
        },
        "rows": out_rows,
        "runtime_seconds": runtime,
    }
    out_path = REPO / "experiments/outputs/backtest_6yr_replay.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n=== 6-year DK1 replay (noisy bidder + {HALLUCINATION_RATE*100:.0f}% hallucinations) ===")
    print(f"Hallucinated bids (unguarded): n={len(hallu_un):,d} "
          f"mean={out['hallucinated_unguarded']['mean']:.2f} "
          f"frac_neg={out['hallucinated_unguarded']['frac_negative']:.3f} "
          f"q05={out['hallucinated_unguarded']['q05']:.2f}")
    print(f"\n{'tau':>6s} {'unguard':>10s} {'guarded':>10s} {'gap':>10s} {'acc%':>6s} {'hallu_filt%':>12s} {'hallu_acc_mean':>15s}")
    for r in out_rows:
        print(f"{r['tau_eur']:6.0f} {r['unguarded_mean_eur']:10.2f} {r['guarded_mean_eur']:10.2f} "
              f"{r['gap_mean_eur']:10.2f} {r['guarded_frac_accepted']*100:6.1f} "
              f"{r['hallu_filter_rate']*100:12.1f} {r['hallu_accepted_mean_profit']:15.2f}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
