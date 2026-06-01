"""F0 — autoregression order-24. Per docs/RESEARCH-PROPOSAL.md §4.2.2 (forecaster zoo).

A literal AR(24) on the quarter-hourly imbalance-price target, fit via OLS in
``statsmodels``. Quantile predictions are produced from the residual standard
error and the in-sample residual quantiles (the latter is more honest than
assuming Gaussianity, but we provide both).

Trained per-seed in the same protocol as F7/F8 so the leaderboard is fair.
The model itself is deterministic given the train window — the seed only
controls residual sampling for the empirical-quantile path.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from statsmodels.tsa.ar_model import AutoReg

from heimdall_forecaster.train.dataset import (
    HORIZON,
    QUANTILES,
    SEQ_LEN,
    TARGET_COL,
    WindowStats,
    make_windows,
)
from heimdall_ml import seeds, tracking


@dataclass
class F0Config:
    name: str = "f0"
    train_panel: Path = Path("data/processed/dk1_panel_train.parquet")
    val_panel: Path = Path("data/processed/dk1_panel_val.parquet")
    seq_len: int = SEQ_LEN
    horizon: int = HORIZON
    ar_order: int = 24  # per proposal §4.2.2 row F0
    seed: int = 42
    quantiles: tuple[float, ...] = field(default_factory=lambda: QUANTILES)
    out_dir: Path = Path("models/forecaster")
    experiment: str = "heimdall-forecaster-f0"
    target: str = "price"
    target_column: str | None = None


def _series(panel_path: Path, target: str = "price", target_column: str | None = None) -> np.ndarray:
    df = pl.read_parquet(panel_path).sort("timestamp_utc")
    if target_column is not None:
        col = target_column
    elif target == "activation_volume":
        # Signed activation volume in MWh (15-min): (up - down) * 0.25
        up = df["mfrr_up_volume_mw"].to_numpy().astype(np.float64)
        dn = df["mfrr_down_volume_mw"].to_numpy().astype(np.float64)
        y = (up - dn) * 0.25
        mask = ~np.isnan(y)
        return y[mask]
    else:
        col = TARGET_COL
    return df.drop_nulls(col)[col].to_numpy().astype(np.float64)


def _fit_ar(y: np.ndarray, *, lags: int) -> AutoReg:
    model = AutoReg(y, lags=lags, old_names=False)
    return model.fit()


def _rolling_predict(
    fitted, history: np.ndarray, *, horizon: int
) -> np.ndarray:
    """Return (N, horizon) point predictions, where N = len(history) - lags."""
    params = fitted.params  # const + lag coefficients
    if fitted.model.trend == "c":
        const = params[0]
        coefs = params[1:]
    else:
        const = 0.0
        coefs = params
    lags = coefs.size
    n = len(history) - lags
    if n <= 0:
        raise ValueError(f"history too short for lags={lags}")
    out = np.empty((n, horizon), dtype=np.float64)
    for i in range(n):
        window = history[i : i + lags].copy()
        for h in range(horizon):
            yhat = const + coefs @ window[::-1]  # AR(p) standard convention
            out[i, h] = yhat
            # Roll the window forward with the new prediction.
            window = np.concatenate([window[1:], [yhat]])
    return out


def _quantiles_from_residuals(
    point: np.ndarray, residuals: np.ndarray, levels: Iterable[float]
) -> np.ndarray:
    qs = np.quantile(residuals, list(levels))
    out = np.empty((*point.shape, len(qs)), dtype=np.float64)
    for i, q in enumerate(qs):
        out[..., i] = point + q
    return out


def train_f0(cfg: F0Config) -> dict[str, object]:
    seeds.seed_everything(cfg.seed)
    y_tr = _series(cfg.train_panel, target=cfg.target, target_column=cfg.target_column)
    y_va = _series(cfg.val_panel, target=cfg.target, target_column=cfg.target_column)
    fitted = _fit_ar(y_tr, lags=cfg.ar_order)

    # In-sample residuals = y_t - yhat_t (t > p).
    yhat_in = fitted.predict(start=cfg.ar_order, end=len(y_tr) - 1)
    residuals = y_tr[cfg.ar_order :] - yhat_in

    # Build val targets matching make_windows shape (so the leaderboard is fair).
    _, Y_va, _stats = make_windows(
        cfg.val_panel,
        seq_len=cfg.seq_len,
        horizon=cfg.horizon,
        multivariate=False,
    )

    # Predict horizon-ahead from each val window's last `ar_order` points.
    # We re-build the rolling history vector by concatenating the prefix from
    # the train series and the val series, then sliding.
    full = np.concatenate([y_tr, y_va])
    # The first val target window starts at index len(y_tr) - cfg.seq_len + cfg.seq_len = len(y_tr).
    # Forecast start indices map to historical anchors len(y_tr) - cfg.ar_order, ..., etc.
    n_val_windows = Y_va.shape[0]
    point = np.empty((n_val_windows, cfg.horizon), dtype=np.float64)
    for i in range(n_val_windows):
        anchor = len(y_tr) + i  # first target ts of window i
        history = full[anchor - cfg.ar_order : anchor]
        # Rolling forecast.
        params = fitted.params
        const = params[0] if fitted.model.trend == "c" else 0.0
        coefs = params[1:] if fitted.model.trend == "c" else params
        window = history.copy()
        for h in range(cfg.horizon):
            yhat = const + coefs @ window[::-1]
            point[i, h] = yhat
            window = np.concatenate([window[1:], [yhat]])

    # Empirical quantiles from residuals.
    quantile_offsets = np.quantile(residuals, list(cfg.quantiles))
    val_preds = np.empty((n_val_windows, cfg.horizon, len(cfg.quantiles)), dtype=np.float64)
    for qi, off in enumerate(quantile_offsets):
        val_preds[:, :, qi] = point + off

    # Y_va is in normalised units (make_windows normalised target with stats).
    Y_va_dn = _stats.denormalise_target(Y_va)
    # val_preds and Y_va_dn now both in DKK/MWh.

    per_q = {}
    for qi, q in enumerate(cfg.quantiles):
        err = Y_va_dn - val_preds[:, :, qi]
        per_q[f"val_pinball_q{int(q*100)}"] = float(
            np.mean(np.maximum(q * err, (q - 1.0) * err))
        )
    per_q["val_pinball_mean_dkk"] = float(np.mean(list(per_q.values())))

    sorted_pred = np.sort(val_preds, axis=-1)
    coverage = float(
        np.mean(
            (Y_va_dn >= sorted_pred[:, :, 0]) & (Y_va_dn <= sorted_pred[:, :, -1])
        )
    )

    out_dir = cfg.out_dir / cfg.name / f"seed-{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracking.init(experiment=cfg.experiment)
    params = {
        "name": cfg.name,
        "seed": cfg.seed,
        "ar_order": cfg.ar_order,
        "seq_len": cfg.seq_len,
        "horizon": cfg.horizon,
        "n_train": int(len(y_tr)),
        "n_val_windows": int(n_val_windows),
    }
    with tracking.run(name=f"{cfg.name}-seed{cfg.seed}", params=params):
        tracking.log_metrics(per_q)
        tracking.log_metrics({"val_q10_q90_coverage": coverage})

    np.savez(out_dir / "val_preds.npz", preds=val_preds, targets=Y_va_dn)
    with open(out_dir / "config.json", "w") as fh:
        json.dump(params, fh, indent=2)
    with open(out_dir / "stats.pkl", "wb") as fh:
        pickle.dump(_stats, fh)
    # AR coefficients persistence (for reproducibility)
    np.savez(
        out_dir / "model.npz",
        params=fitted.params,
        residual_quantiles=quantile_offsets,
        residual_std=float(residuals.std()),
        ar_order=cfg.ar_order,
    )
    metrics = {**per_q, "val_q10_q90_coverage": coverage}
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    return {
        "val_pinball_mean": per_q["val_pinball_mean_dkk"],
        "val_q10_q90_coverage": coverage,
        "per_quantile": per_q,
        "ckpt": out_dir / "model.npz",
        "val_preds": val_preds,
        "val_targets": Y_va_dn,
    }


__all__ = ["F0Config", "train_f0"]
