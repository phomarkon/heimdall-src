"""Evaluate F10 (Chronos-Bolt zero-shot) on the post-break val window.

Produces a directly-comparable val-pinball number for the F-zoo
leaderboard, alongside ACI marginal coverage.  Chronos-Bolt is
deterministic given inputs (no seed-randomness in the forward pass)
so the 5-seed mean is just five identical runs; we report it for
consistency with the proposal §5.3.1 protocol.

Usage:
  PYTHONPATH=. python experiments/eval_f10_chronos_bolt.py \
      --model amazon/chronos-bolt-tiny --n-windows 200

Outputs:
  models/forecaster/f10/seed-<seed>/{val_preds.npz, metrics.json,
                                     config.json}
  notes/forecaster_leaderboard.md  (re-rendered)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
QUANTILES = (0.1, 0.5, 0.9)
SEQ_LEN = 192          # 2 days of 15-min ticks; matches F7
HORIZON = 16           # 4 hours; matches F7
FROZEN_SEEDS = (13, 42, 137, 1729, 31415)


def _pinball(y: np.ndarray, q: np.ndarray, level: float) -> float:
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def _make_windows(series: np.ndarray, n_windows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = series.size - SEQ_LEN - HORIZON
    if n_windows is not None:
        n = min(n, n_windows)
    X = np.empty((n, SEQ_LEN), dtype=np.float64)
    Y = np.empty((n, HORIZON), dtype=np.float64)
    for i in range(n):
        X[i] = series[i : i + SEQ_LEN]
        Y[i] = series[i + SEQ_LEN : i + SEQ_LEN + HORIZON]
    return X, Y


def evaluate(model_id: str = "amazon/chronos-bolt-tiny", n_windows: int | None = None,
             batch_size: int = 64, seed: int = 42) -> dict:
    import torch
    from chronos import BaseChronosPipeline

    df = pl.read_parquet(REPO_ROOT / "data/processed/dk1_panel_val.parquet").sort("timestamp_utc")
    series = df["imbalance_price_dkk_mwh_15min"].drop_nulls().to_numpy().astype(np.float64)

    X, Y = _make_windows(series, n_windows)
    n = X.shape[0]

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=device_map, torch_dtype=torch.float32)

    preds = np.empty((n, HORIZON, len(QUANTILES)), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(0, n, batch_size):
        ctx = torch.tensor(X[i : i + batch_size], dtype=torch.float32)
        q, _ = pipe.predict_quantiles(
            inputs=ctx, prediction_length=HORIZON, quantile_levels=list(QUANTILES),
        )
        preds[i : i + batch_size] = q.cpu().numpy()
    runtime = time.perf_counter() - t0

    out_dir = REPO_ROOT / "models/forecaster/f10" / f"seed-{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "val_preds.npz", preds=preds, targets=Y)

    per_q = {f"val_pinball_q{int(qq*100)}": _pinball(Y, preds[..., qi], qq)
             for qi, qq in enumerate(QUANTILES)}
    pinball_mean = float(np.mean(list(per_q.values())))
    sorted_p = np.sort(preds, axis=-1)
    coverage = float(np.mean((Y >= sorted_p[..., 0]) & (Y <= sorted_p[..., -1])))

    metrics = {
        "seed": seed,
        "model_id": model_id,
        "n_windows": n,
        "runtime_seconds": round(runtime, 2),
        **per_q,
        "val_pinball_mean_dkk": pinball_mean,
        "val_q10_q90_coverage": coverage,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "config.json").write_text(json.dumps({
        "model_id": model_id, "seq_len": SEQ_LEN, "horizon": HORIZON,
        "quantiles": list(QUANTILES),
    }, indent=2))
    return metrics


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="amazon/chronos-bolt-tiny")
    p.add_argument("--n-windows", type=int, default=200,
                   help="Cap windows for a fast leaderboard cell; full=~5000")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="*", default=[42],
                   help="Chronos is deterministic; we record one seed by default")
    args = p.parse_args()
    summary = []
    for seed in args.seeds:
        m = evaluate(args.model, args.n_windows, args.batch_size, seed)
        summary.append(m)
        print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
