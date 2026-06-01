"""F10 (Chronos-Bolt) TEST-set evaluation in a side-venv.

Mirrors run_f10_eval.py but evaluates on dk1_panel_test.parquet and
writes the per-seed NPZ to experiments/outputs/test_preds/f10/, matching
the layout produced by experiments/test_set_evaluation.py for other
backends. Used to generate the F10 horizon-decay appendix figure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from chronos import BaseChronosPipeline

REPO = Path(__file__).resolve().parents[3]
TEST_PANEL = REPO / "data" / "processed" / "dk1_panel_test.parquet"
PREDS_DIR = REPO / "experiments" / "outputs" / "test_preds" / "f10"

SEQ_LEN = 192
HORIZON = 16
LEVELS = (0.1, 0.5, 0.9)


def _windows(series: np.ndarray, n_eval: int) -> tuple[np.ndarray, np.ndarray]:
    n_max = max(0, series.size - SEQ_LEN - HORIZON)
    n = min(n_eval, n_max)
    contexts = np.stack([series[i : i + SEQ_LEN] for i in range(n)])
    targets = np.stack([series[i + SEQ_LEN : i + SEQ_LEN + HORIZON] for i in range(n)])
    return contexts, targets


def _predict_batch(pipe: BaseChronosPipeline, contexts: np.ndarray, batch: int = 32) -> np.ndarray:
    out = np.empty((contexts.shape[0], HORIZON, 3), dtype=np.float32)
    quantile_idx = None
    for start in range(0, contexts.shape[0], batch):
        chunk = torch.tensor(contexts[start : start + batch], dtype=torch.float32)
        preds = pipe.predict(chunk, prediction_length=HORIZON).cpu().numpy()
        if quantile_idx is None:
            n_q = preds.shape[1]
            qs = np.linspace(0, 1, n_q + 2)[1:-1]
            quantile_idx = [int(np.argmin(np.abs(qs - lv))) for lv in LEVELS]
        out[start : start + chunk.shape[0]] = preds[:, quantile_idx, :].transpose(0, 2, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--model", default="amazon/chronos-bolt-tiny")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[f10-test] loading {args.model} on {args.device} ...", flush=True)
    pipe = BaseChronosPipeline.from_pretrained(args.model, device_map=args.device)

    df = pl.read_parquet(TEST_PANEL).sort("timestamp_utc")
    series = df["imbalance_price_dkk_mwh_15min"].drop_nulls().to_numpy().astype(np.float32)
    contexts, targets = _windows(series, args.n_test)
    print(f"[f10-test] test: {contexts.shape[0]} windows", flush=True)

    preds = _predict_batch(pipe, contexts)

    out = PREDS_DIR / f"seed-{args.seed}.npz"
    np.savez(out, preds=preds, targets=targets.astype(np.float32))
    print(f"[f10-test] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
