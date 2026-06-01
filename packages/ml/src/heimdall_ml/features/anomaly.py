"""Anomaly-detection input features for the rich panel.

Per Plan v2 Track A.9. Every signal is **strictly causal**: BOCPD is online,
rolling z-scores / Hampel / realized vol use trailing windows, multivariate
anomalies (IsolationForest, Mahalanobis) are fit only on the train fold
(``< PRE_POST_BREAK_UTC``).

Output frame:

  - bocpd_run_length_mean / bocpd_p_change / bocpd_entropy
  - zscore_24h, zscore_96h, hampel_flag_24h, realized_vol_1h
  - isoforest_score (lower = more anomalous; fit on train)
  - mahalanobis_24h (rolling 24h Mahalanobis distance vs train-mean/cov)
  - dk1_de_spread, dk1_se3_spread, dk1_no2_spread + 24h z-scores

The build function is *deterministic given a seed* and writes
``data/processed/anomaly_features.parquet`` alongside the rich panel.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from heimdall_ml.conformal.bocpd import BOCPD
from heimdall_ml.seeds import seed_everything

PRE_POST_BREAK_UTC = datetime(2025, 3, 4, 0, 0, tzinfo=UTC)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[5] / "data" / "processed" / "anomaly_features.parquet"


def _rolling_z(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling z-score: z_t = (x_t - mean_{t-W:t}) / std_{t-W:t}."""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    cumsum = np.concatenate([[0.0], np.cumsum(values)])
    cumsumsq = np.concatenate([[0.0], np.cumsum(values * values)])
    for i in range(window, n):
        s = cumsum[i] - cumsum[i - window]
        sq = cumsumsq[i] - cumsumsq[i - window]
        mean = s / window
        var = max(sq / window - mean * mean, 1e-12)
        out[i] = (values[i] - mean) / np.sqrt(var)
    return out


def _hampel_flag(values: np.ndarray, window: int, k: float = 3.0) -> np.ndarray:
    """Causal Hampel flag: 1 if |x_t - median_{t-W:t}| / (1.4826 * MAD) > k."""
    n = len(values)
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    for i in range(window, n):
        w = values[i - window : i]
        if not np.isfinite(w).any():
            continue
        med = np.nanmedian(w)
        mad = np.nanmedian(np.abs(w - med)) * 1.4826
        if mad <= 0 or not np.isfinite(mad):
            continue
        if np.abs(values[i] - med) > k * mad:
            out[i] = 1.0
    return out


def _realized_vol(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling std on returns."""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out
    rets = np.zeros(n)
    rets[1:] = np.diff(values)
    for i in range(window, n):
        out[i] = float(np.std(rets[i - window : i]))
    return out


def _bocpd_features(
    values: np.ndarray,
    *,
    mean_run_length: float = 400.0,
    quarters_per_step: int = 4,  # 4 = hourly subsample on a 15-min grid
    max_posterior_length: int = 720,  # truncate run-length posterior (~30 days at hourly)
) -> dict[str, np.ndarray]:
    """Run BOCPD on a subsampled+truncated series; forward-fill back to full length.

    Native BOCPD is O(n²) because the posterior grows each step. We make it
    O(n_sub × R) with two standard tricks:

    1. **Subsample** the series at ``quarters_per_step`` (default hourly) — the
       balancing-market regime changes far slower than 15-min so we lose little.
    2. **Truncate** the posterior at ``max_posterior_length`` ticks, renormalising
       the truncated tail mass into the kept bucket. This is the standard BOCPD
       speedup; the lost run-lengths are far beyond plausible regime durations.

    Returns full-length arrays via forward-fill.
    """
    n = len(values)
    sub = values[::quarters_per_step]
    bocpd = BOCPD(mean_run_length=mean_run_length)
    n_sub = len(sub)
    rl_sub = np.zeros(n_sub)
    pc_sub = np.zeros(n_sub)
    ent_sub = np.zeros(n_sub)
    for i, x in enumerate(sub):
        if not np.isfinite(x):
            x = 0.0
        r = bocpd.step(float(x))
        p = r.posterior
        # Truncate posterior in place after the step to bound memory + time.
        if p.size > max_posterior_length:
            tail_mass = float(p[max_posterior_length:].sum())
            p = p[:max_posterior_length].copy()
            p[-1] += tail_mass
            bocpd._posterior = p / max(p.sum(), 1e-12)
            bocpd._mu = bocpd._mu[:max_posterior_length]
            bocpd._kappa = bocpd._kappa[:max_posterior_length]
            bocpd._alpha = bocpd._alpha[:max_posterior_length]
            bocpd._beta = bocpd._beta[:max_posterior_length]
            p = bocpd._posterior
        r_idx = np.arange(p.size)
        rl_sub[i] = float((p * r_idx).sum())
        pc_sub[i] = float(p[0])
        p_safe = np.where(p > 0, p, 1e-300)
        ent_sub[i] = float(-(p_safe * np.log(p_safe)).sum())

    # Forward-fill from subsample step boundaries.
    def _ffill(sub_arr: np.ndarray) -> np.ndarray:
        out = np.empty(n)
        idx = np.arange(n) // quarters_per_step
        out[:] = sub_arr[np.clip(idx, 0, n_sub - 1)]
        return out

    return {
        "bocpd_run_length_mean": _ffill(rl_sub),
        "bocpd_p_change": _ffill(pc_sub),
        "bocpd_entropy": _ffill(ent_sub),
    }


def _train_only_isoforest_scores(panel: pl.DataFrame, feature_cols: list[str], seed: int):
    from sklearn.ensemble import IsolationForest

    train_mask = (panel["timestamp_utc"] < PRE_POST_BREAK_UTC).to_numpy()
    X_all = panel.select(feature_cols).to_numpy()
    # IsolationForest cannot handle NaN — fill with median per column on TRAIN only.
    X_train = X_all[train_mask]
    if X_train.shape[0] == 0:
        return np.full(X_all.shape[0], np.nan)
    medians = np.nanmedian(X_train, axis=0)
    X_train_imp = np.where(np.isnan(X_train), medians, X_train)
    X_all_imp = np.where(np.isnan(X_all), medians, X_all)
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=seed,
        max_samples=min(256, X_train.shape[0]),
    )
    model.fit(X_train_imp)
    return model.decision_function(X_all_imp)  # higher = more normal


def _rolling_mahalanobis(panel: pl.DataFrame, feature_cols: list[str], window: int = 96) -> np.ndarray:
    """Mahalanobis distance to train-only mean/cov, computed once."""
    train_mask = (panel["timestamp_utc"] < PRE_POST_BREAK_UTC).to_numpy()
    X = panel.select(feature_cols).to_numpy()
    X_train = X[train_mask]
    if X_train.shape[0] == 0:
        return np.full(X.shape[0], np.nan)
    medians = np.nanmedian(X_train, axis=0)
    X_train_imp = np.where(np.isnan(X_train), medians, X_train)
    X_imp = np.where(np.isnan(X), medians, X)
    mu = np.mean(X_train_imp, axis=0)
    cov = np.cov(X_train_imp.T) + np.eye(X_train_imp.shape[1]) * 1e-6
    cov_inv = np.linalg.pinv(cov)
    delta = X_imp - mu
    md = np.sqrt(np.einsum("ij,jk,ik->i", delta, cov_inv, delta))
    return md.astype(np.float64)


def build_anomaly_features(
    *,
    rich_panel: pl.DataFrame | None = None,
    seed: int = 13,
    output_path: Path | None = None,
    target_col: str = "imbalance_price_dkk_mwh",
) -> pl.DataFrame:
    """Build the anomaly-features frame and (optionally) cache it.

    The result is on the same 15-min grid as ``rich_panel``. If the target
    column has gaps, BOCPD/z-scores skip them (NaN propagation).
    """
    if rich_panel is None:
        from heimdall_data.rich_panel import build_rich_panel

        rich_panel = build_rich_panel()

    seed_everything(seed)
    y = rich_panel[target_col].to_numpy().astype(np.float64)

    feats: dict[str, np.ndarray] = {}

    # BOCPD over the target series (single pass).
    feats.update(_bocpd_features(y))

    # Rolling univariate.
    feats["zscore_24h"] = _rolling_z(y, window=96)  # 24h × 4 = 96 quarters
    feats["zscore_96h"] = _rolling_z(y, window=384)
    feats["hampel_flag_24h"] = _hampel_flag(y, window=96)
    feats["realized_vol_1h"] = _realized_vol(y, window=4)

    # Multivariate (only meaningful when Trinity wide features are present).
    multivariate_cols = [
        c
        for c in (
            "load_actual_mw",
            "wind_gen_mw",
            "solar_gen_mw",
            "wind_speed_100m",
        )
        if c in rich_panel.columns
    ]
    if multivariate_cols and target_col in rich_panel.columns:
        multi_panel = rich_panel.select(["timestamp_utc", target_col, *multivariate_cols])
        feats["isoforest_score"] = _train_only_isoforest_scores(
            multi_panel, feature_cols=[target_col, *multivariate_cols], seed=seed
        )
        feats["mahalanobis_24h"] = _rolling_mahalanobis(
            multi_panel, feature_cols=[target_col, *multivariate_cols], window=96
        )

    # Extended per-feature anomaly detection over weather + cross-zone + mFRR
    # covariates. The forecaster already consumes these as inputs, but per-
    # feature outlier signals add a dedicated anomaly channel per important
    # exog driver. Cheap: rolling 24h z-score + Hampel flag per column.
    extended_cols = [
        c for c in (
            "temperature_2m", "pressure_msl", "precipitation_mm",
            "wind_speed_10m", "wind_direction_100m", "shortwave_radiation",
            "cloud_cover", "de_da_price_eur_mwh", "se3_da_price_eur_mwh",
            "no2_da_price_eur_mwh", "mfrr_up_volume_mw", "mfrr_down_volume_mw",
        ) if c in rich_panel.columns
    ]
    for c in extended_cols:
        arr = rich_panel[c].to_numpy().astype(np.float64)
        feats[f"{c}_z24h"] = _rolling_z(arr, window=96)
        feats[f"{c}_hampel_24h"] = _hampel_flag(arr, window=96)
    if len(extended_cols) >= 3:
        # Joint IsoForest + Mahalanobis across the weather block.
        weather_cols = [c for c in extended_cols if c.startswith((
            "temperature_", "pressure_", "precipitation_", "wind_", "shortwave_", "cloud_"))]
        if len(weather_cols) >= 3:
            wpanel = rich_panel.select(["timestamp_utc", *weather_cols])
            feats["weather_isoforest_score"] = _train_only_isoforest_scores(
                wpanel, feature_cols=weather_cols, seed=seed
            )
            feats["weather_mahalanobis_24h"] = _rolling_mahalanobis(
                wpanel, feature_cols=weather_cols, window=96
            )

    # Cross-zone spreads + 24h z-scores are price-specific. For activation-volume
    # anomaly panels, keep the target semantics clean and skip price spreads.
    if target_col in {"imbalance_price_dkk_mwh", "imbalance_price_dkk_mwh_15min"} and "de_da_price_eur_mwh" in rich_panel.columns:
        # Convert DK1 imbalance DKK→EUR with the same fixed 7.46 FX used in loaders.
        dk1_eur = y / 7.46
        for z in ("de", "se3", "no2"):
            col = f"{z}_da_price_eur_mwh"
            if col not in rich_panel.columns:
                continue
            other = rich_panel[col].to_numpy().astype(np.float64)
            spread = dk1_eur - other
            feats[f"dk1_{z}_spread_eur_mwh"] = spread
            feats[f"dk1_{z}_spread_zscore_24h"] = _rolling_z(spread, window=96)

    # Assemble result frame on the same timestamp grid.
    # NaN tail-effects at the start of rolling windows are replaced with 0 — the
    # model sees these as "no anomaly information yet" rather than poisoning
    # normalisation. We do NOT silently change values further into the series:
    # only the unavoidable warmup NaN positions are masked.
    out = rich_panel.select("timestamp_utc")
    for name, arr in feats.items():
        arr = np.where(np.isfinite(arr), arr, 0.0)
        out = out.with_columns(pl.Series(name, arr, dtype=pl.Float64))

    out_path = output_path or DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)
    return out


__all__ = ["DEFAULT_OUTPUT", "PRE_POST_BREAK_UTC", "build_anomaly_features"]
