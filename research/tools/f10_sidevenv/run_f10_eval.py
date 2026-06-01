"""F10 (Chronos-Bolt) val + test evaluation in a side-venv.

The chronos-forecasting==1.5.2 package transitively requires
huggingface-hub<1.0 and transformers<4.55, which conflicts with the main
Heimdall env (HF >=1.14 for the HF model-card mirror). This script runs
inside a separate uv-managed venv at tools/f10_sidevenv/.venv and exports
val_preds.npz back into models/forecaster/f10/seed-<seed>/ for downstream
consumption by the main env (CalibratedForecaster, multi-α ACI, etc.).

Run:
    source tools/f10_sidevenv/.venv/bin/activate
    python tools/f10_sidevenv/run_f10_eval.py --seed 13
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from chronos import BaseChronosPipeline

REPO = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO / "models" / "forecaster" / "f10"
VAL_PANEL = REPO / "data" / "processed" / "dk1_panel_val.parquet"
TEST_PANEL = REPO / "data" / "processed" / "dk1_panel_test.parquet"

SEQ_LEN = 192
HORIZON = 16
LEVELS = (0.1, 0.5, 0.9)


def _windows(series: np.ndarray, n_eval: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (n_eval, SEQ_LEN) context and (n_eval, HORIZON) target arrays."""
    n_max = max(0, series.size - SEQ_LEN - HORIZON)
    n = min(n_eval, n_max)
    contexts = np.stack([series[i : i + SEQ_LEN] for i in range(n)])
    targets = np.stack([series[i + SEQ_LEN : i + SEQ_LEN + HORIZON] for i in range(n)])
    return contexts, targets


def _predict_batch(pipe: BaseChronosPipeline, contexts: np.ndarray, batch: int = 32) -> np.ndarray:
    """Predict quantiles for many windows. Returns (n, H, 3) in DKK/MWh."""
    out = np.empty((contexts.shape[0], HORIZON, 3), dtype=np.float32)
    quantile_idx = None
    for start in range(0, contexts.shape[0], batch):
        chunk = torch.tensor(contexts[start : start + batch], dtype=torch.float32)
        # chronos-bolt returns (B, num_quantiles, H); 9 levels by default
        preds = pipe.predict(chunk, prediction_length=HORIZON).cpu().numpy()
        if quantile_idx is None:
            n_q = preds.shape[1]
            qs = np.linspace(0, 1, n_q + 2)[1:-1]
            quantile_idx = [int(np.argmin(np.abs(qs - lv))) for lv in LEVELS]
        out[start : start + chunk.shape[0]] = preds[:, quantile_idx, :].transpose(0, 2, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--model", default="amazon/chronos-bolt-base")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    out_dir = MODEL_DIR / f"seed-{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[f10] loading {args.model} on {args.device} ...", flush=True)
    pipe = BaseChronosPipeline.from_pretrained(args.model, device_map=args.device)

    df = pl.read_parquet(VAL_PANEL).sort("timestamp_utc")
    series = df["imbalance_price_dkk_mwh_15min"].drop_nulls().to_numpy().astype(np.float32)
    contexts, targets = _windows(series, args.n_val)
    print(f"[f10] val: {contexts.shape[0]} windows", flush=True)

    preds = _predict_batch(pipe, contexts)  # (n, H, 3)
    # save
    np.savez(out_dir / "val_preds.npz", preds=preds, targets=targets.astype(np.float32))

    pin = []
    for qi, lv in enumerate(LEVELS):
        err = targets - preds[..., qi]
        pin.append(float(np.mean(np.maximum(lv * err, (lv - 1.0) * err))))
    raw_cov = float(np.mean((targets >= preds[..., 0]) & (targets <= preds[..., -1])))
    pinball_mean = float(np.mean(pin))

    metrics = {
        "seed": args.seed,
        "val_pinball_q10": pin[0],
        "val_pinball_q50": pin[1],
        "val_pinball_q90": pin[2],
        "val_pinball_mean": pinball_mean,
        "val_q10_q90_coverage": raw_cov,
        "n_val_windows": int(contexts.shape[0]),
        "model_name": "f10",
        "method": f"Chronos-Bolt zero-shot ({args.model})",
        "_source": "tools/f10_sidevenv/run_f10_eval.py",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    config = {
        "name": "f10",
        "seed": args.seed,
        "model": args.model,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "quantiles": list(LEVELS),
        "side_venv": True,
        "out_dir": str(out_dir),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"[f10] seed={args.seed} pinball={pinball_mean:.1f} raw_cov={raw_cov:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
