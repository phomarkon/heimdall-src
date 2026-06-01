"""F9 (TimesFM-2.0) leaderboard-compatible eval.

Produces the same canonical artifacts as F7/F8/F11:
  models/forecaster/f9/seed-<s>/{val_preds.npz, metrics.json, config.json}

shape preds = (N, H, Q) with Q = (0.1, 0.5, 0.9) — Q-quantiles mapped from
TimesFM's 9 native deciles to nearest decile. Targets shape (N, H).

TimesFM is deterministic at inference; the 5-seed sweep is recorded for
zoo consistency (per proposal §5.3.1) — values across seeds are identical.

Usage:
    PYTHONPATH=. python experiments/eval_f9_timesfm_zoo.py [--n-windows N]
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
SEQ_LEN = 192
HORIZON = 16
QUANTILES = (0.1, 0.5, 0.9)
FROZEN_SEEDS = (13, 42, 137, 1729, 31415)
# TimesFM emits deciles 0.1..0.9 (indices 0..8 of `full[:, 1:]`).
DECILE_IDX_FOR_Q = {0.1: 0, 0.5: 4, 0.9: 8}


def _pinball(y, q, level):
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def _make_windows(series, n_windows=None):
    n = series.size - SEQ_LEN - HORIZON
    if n_windows is not None:
        n = min(n, n_windows)
    X = np.empty((n, SEQ_LEN), dtype=np.float64)
    Y = np.empty((n, HORIZON), dtype=np.float64)
    for i in range(n):
        X[i] = series[i : i + SEQ_LEN]
        Y[i] = series[i + SEQ_LEN : i + SEQ_LEN + HORIZON]
    return X, Y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=None,
                    help="cap; None = full val (~5360 windows)")
    ap.add_argument("--backend", default="gpu")
    args = ap.parse_args()

    from heimdall_forecaster.timesfm_wrapper import TimesFMForecaster

    df = pl.read_parquet(REPO / "data/processed/dk1_panel_val.parquet").sort("timestamp_utc")
    y = df["imbalance_price_dkk_mwh_15min"].drop_nulls().to_numpy().astype(np.float64)
    X, Y = _make_windows(y, args.n_windows)
    n = X.shape[0]
    print(f"[f9] windows={n}, seq={SEQ_LEN}, horizon={HORIZON}")

    f9 = TimesFMForecaster(
        backend=args.backend, context_len=SEQ_LEN, horizon_len=HORIZON
    )
    f9._load()

    preds = np.empty((n, HORIZON, len(QUANTILES)), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(n):
        mean, full = f9.predict(X[i])
        # full shape: (horizon, 10) — col 0 mean/median, cols 1..9 deciles
        for qi, q in enumerate(QUANTILES):
            preds[i, :, qi] = full[:, 1 + DECILE_IDX_FOR_Q[q]]
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{n}]  elapsed={time.perf_counter()-t0:.1f}s")
    runtime = time.perf_counter() - t0
    print(f"[f9] inference done in {runtime:.1f}s")

    per_q = {f"val_pinball_q{int(qq*100)}": _pinball(Y, preds[..., qi], qq)
             for qi, qq in enumerate(QUANTILES)}
    pinball_mean = float(np.mean(list(per_q.values())))
    sorted_p = np.sort(preds, axis=-1)
    coverage = float(np.mean((Y >= sorted_p[..., 0]) & (Y <= sorted_p[..., -1])))

    # ACI wrap so the Theorem 1b panel is honest for F9.
    # Save an intermediate val_preds.npz to disk first, then call the shared wrapper.
    from heimdall_forecaster.train.wrap_aci import aci_coverage_from_val
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez(tmp.name, preds=preds, targets=Y)
        aci = aci_coverage_from_val(tmp.name, alpha=0.1, gamma=0.05)
    aci_payload = {
        "aci_alpha_target": aci.alpha_target,
        "aci_empirical_coverage": aci.empirical_coverage,
        "aci_mean_width": aci.mean_width,
    }

    # Save same artifacts under all 5 frozen seeds (deterministic).
    for seed in FROZEN_SEEDS:
        out_dir = REPO / "models/forecaster/f9" / f"seed-{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "val_preds.npz", preds=preds, targets=Y)
        metrics = {
            "seed": seed,
            "model_id": "google/timesfm-2.0-500m-pytorch",
            "n_windows": n,
            "runtime_seconds": round(runtime, 2),
            **per_q,
            "val_pinball_mean_dkk": pinball_mean,
            "val_pinball_mean": pinball_mean,  # leaderboard alias
            "val_q10_q90_coverage": coverage,
            **aci_payload,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (out_dir / "config.json").write_text(json.dumps({
            "model_id": "google/timesfm-2.0-500m-pytorch",
            "seq_len": SEQ_LEN, "horizon": HORIZON, "quantiles": list(QUANTILES),
            "deterministic": True, "seed_recorded_for_parity": seed,
        }, indent=2))
    print(json.dumps({
        "n_windows": n, "runtime_s": round(runtime, 2),
        "val_pinball_mean": pinball_mean, "val_q10_q90_coverage": coverage,
        **per_q,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
