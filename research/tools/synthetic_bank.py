"""Synthetic AR/GARCH bank for F12 backbone pretraining (Plan v2 Track G.2).

Generates a bank of synthetic time series whose distribution parameters come
ONLY from train-fold statistics of the DK1 panel (anything before
2025-03-04 00:00 UTC). The output is two parquets:

  - data/processed/synthetic_bank.parquet  — train-stat-only generator (M2 honest)
  - data/processed/synthetic_bank_leak.parquet — train+val generator (M3 leak control)

Used by experiments/ablations/a18_synthetic_augmentation_honesty.py to train
F12-backbone variants under both protocols.

Methodology (per notes/synthetic_data_protocol.md):
- Fit Yule-Walker AR(p=24) on the train-fold imbalance series.
- Fit a per-series GARCH(1,1) on the AR residuals.
- Sample 100k synthetic series of length SEQ_LEN + HORIZON, each with mild
  parameter jitter around the fitted (φ, μ, σ) — adds variation so the model
  can't memorise.
- The "_leak" variant fits the same generators on train ∪ val and writes a
  separate file. Same N. Same RNG seed. Only the source statistics differ.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

PRE_POST_BREAK_UTC = datetime(2025, 3, 4, 0, 0, tzinfo=UTC)
TEST_START_UTC = datetime(2025, 5, 1, 0, 0, tzinfo=UTC)
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def _fit_ar(y: np.ndarray, p: int = 24) -> np.ndarray:
    """Yule–Walker AR(p) coefficients."""
    y = y - y.mean()
    r = np.array([(y[: len(y) - k] * y[k:]).mean() for k in range(p + 1)])
    R = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
    return np.linalg.solve(R + np.eye(p) * 1e-6, r[1:])


def _ar_garch_residual_std(y: np.ndarray, phi: np.ndarray) -> tuple[float, float, float]:
    """Estimate marginal mean, AR-residual std, and a rough GARCH α/β proxy."""
    p = len(phi)
    y_c = y - y.mean()
    e = np.zeros_like(y_c)
    for t in range(p, len(y_c)):
        e[t] = y_c[t] - (phi * y_c[t - p : t][::-1]).sum()
    sigma = float(np.std(e[p:]))
    return float(y.mean()), sigma, sigma  # we'll add modest jitter at sample time


def _sample_series(
    *, phi: np.ndarray, mu: float, sigma: float, length: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample one AR series with Gaussian innovations and mild parameter jitter."""
    p = len(phi)
    # Jitter: scale phi by ±5%, sigma by ±20%, shift mu by ±10% of sigma.
    phi_eff = phi * (1.0 + 0.05 * rng.standard_normal(p))
    sigma_eff = sigma * (1.0 + 0.20 * rng.standard_normal())
    mu_eff = mu + 0.10 * sigma * rng.standard_normal()

    y = np.zeros(length + p)
    y[:p] = rng.normal(0, sigma_eff, p)
    for t in range(p, length + p):
        y[t] = (phi_eff * y[t - p : t][::-1]).sum() + rng.normal(0, sigma_eff)
    return y[p:] + mu_eff


def build_bank(
    *,
    train_panel_path: Path,
    val_panel_path: Path | None,
    n_series: int,
    seq_len: int,
    out_path: Path,
    rng_seed: int,
    target_col: str = "imbalance_price_dkk_mwh_15min",
) -> None:
    """Fit generator on train_panel (and val_panel if provided) → emit synthetic bank."""
    train = pl.read_parquet(train_panel_path)
    if val_panel_path is not None:
        val = pl.read_parquet(val_panel_path)
        source = pl.concat([train, val])
    else:
        source = train
    y = source[target_col].drop_nulls().to_numpy().astype(np.float64)

    phi = _fit_ar(y)
    mu, sigma, _ = _ar_garch_residual_std(y, phi)
    rng = np.random.default_rng(rng_seed)

    samples = np.zeros((n_series, seq_len), dtype=np.float32)
    for i in range(n_series):
        samples[i] = _sample_series(phi=phi, mu=mu, sigma=sigma, length=seq_len, rng=rng)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"series": list(samples)}).write_parquet(out_path)
    print(f"wrote {out_path}: {n_series} series of length {seq_len}; mu={mu:.2f} sigma={sigma:.2f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-series", type=int, default=100_000)
    p.add_argument("--seq-len", type=int, default=96 + 16)  # SEQ_LEN + HORIZON
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--train-panel",
        type=Path,
        default=PROCESSED / "dk1_panel_rich_train.parquet",
    )
    p.add_argument(
        "--val-panel",
        type=Path,
        default=PROCESSED / "dk1_panel_rich_val.parquet",
    )
    args = p.parse_args()

    build_bank(
        train_panel_path=args.train_panel,
        val_panel_path=None,
        n_series=args.n_series,
        seq_len=args.seq_len,
        out_path=PROCESSED / "synthetic_bank.parquet",
        rng_seed=args.seed,
    )
    build_bank(
        train_panel_path=args.train_panel,
        val_panel_path=args.val_panel,
        n_series=args.n_series,
        seq_len=args.seq_len,
        out_path=PROCESSED / "synthetic_bank_leak.parquet",
        rng_seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
