"""Sliding-window dataset for F7/F8 training. Per docs/RESEARCH-PROPOSAL.md §4.4.

Sequence length = 96 (24 h × 15-min); horizon = 16 (4 h × 15-min). The default
target is imbalance price, normalised by training-window mean/std. Advisory
activation-volume and activation-direction targets share the same leakage-safe
windowing path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

# Per proposal §4.4
SEQ_LEN = 96
HORIZON = 16
QUANTILES = (0.1, 0.5, 0.9)
TARGET_COL = "imbalance_price_dkk_mwh_15min"
TargetKind = str
F8_FEATURES = ("imbalance_price_dkk_mwh_15min", "load_actual_mw", "da_price_dkk_mwh")
"""F8a (locked baseline) — proposal-faithful 3-feature multivariate stack."""

F8B_FEATURES = F8_FEATURES + (
    # Trinity-vendored cross-zone DA prices + DK1 load (lagged-OK actuals).
    "de_da_price_eur_mwh",
    "se3_da_price_eur_mwh",
    "no2_da_price_eur_mwh",
    # Trinity weather (24h-causal-lagged — see trinity.DEFAULT_WEATHER_LAG_HOURS).
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
    "shortwave_radiation",
    "cloud_cover",
    # Trinity generation actuals (lagged-OK — sliding window only touches t-SEQ_LEN..t).
    "wind_gen_mw",
    "solar_gen_mw",
    # Calendar (positional, no leakage).
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_dk_holiday",
    "is_weekend",
)
"""F8b — rich features (Track A.5 of the max-out plan). All causally honest."""

F8C_FEATURES = F8B_FEATURES + (
    # Cross-border flows (lagged-OK actuals).
    "flow_dk1_de_mw",
    "flow_dk1_se3_mw",
    "flow_dk1_no2_mw",
    # UMM outage features (causal by publication; null pre-2024-03).
    "umm_unavailable_capacity_mw",
    "umm_active_event_count",
    # Anomaly-detection features (computed by heimdall_ml.features.anomaly).
    "bocpd_run_length_mean",
    "bocpd_p_change",
    "bocpd_entropy",
    "zscore_24h",
    "zscore_96h",
    "hampel_flag_24h",
    "realized_vol_1h",
    "isoforest_score",
    "mahalanobis_24h",
    "dk1_de_spread_eur_mwh",
    "dk1_se3_spread_eur_mwh",
    "dk1_no2_spread_eur_mwh",
    "dk1_de_spread_zscore_24h",
)
"""F8c — kitchen-sink (Track D.1). Feed to XAI ranking; the keep-set becomes F8d."""

F8E_FEATURES = F8B_FEATURES + (
    # mFRR activation volumes — the direct cause of imbalance prices.
    # These were present in the panel but missing from earlier feature lists.
    "mfrr_up_volume_mw",
    "mfrr_down_volume_mw",
)
"""F8e — F8b + mFRR activation volumes (audit gap from review pass).
Documented separately from F8b so the (F8e − F8b) delta isolates the mFRR
volume contribution from the rest of the rich feature set."""

# F13 — F8e + causal forecast-error anomaly features. Trained on
# dk1_panel_rich_v2.parquet (53 cols). The hypothesis under test: causal
# anomaly indicators (wind / solar / load forecast errors at issue time)
# carry orthogonal signal to the lagged-price + weather + cross-zone DA
# + statistical anomaly z-scores already in F8b/c.
F13_FEATURES = tuple(c for c in F8E_FEATURES if c not in (
    # Dropped 2026-05-17 (Phase 1.2): 100%-null placeholders + 82%-null UMM.
    "gas_ttf_eur_mwh", "eu_ets_eur_t", "coal_api2_usd_t", "brent_usd_bbl",
    "umm_unavailable_capacity_mw", "umm_active_event_count",
)) + (
    # NEW causal forecast-error features (8 cols).
    "fe_offshore_wind_mw",
    "fe_onshore_wind_mw",
    "fe_solar_mw",
    "fe_total_renewable_mw",
    "fe_total_renewable_24h_mean",
    "fe_total_renewable_24h_std",
    "fe_total_renewable_6h_cumsum",
    "fe_total_renewable_zscore_24h",
)
"""F13 — F8e + 8 causal forecast-error features. See
tools/build_forecast_error_features.py."""

# F8D_FEATURES — lean keep-set derived from A16 XAI ranking
# (experiments/outputs/a16.json; union of SHAP-rank≤15 ∪ perm-importance-rank≤15).
F8D_FEATURES = (
    "imbalance_price_dkk_mwh_15min",
    "da_price_dkk_mwh",
    "load_actual_mw",
    "wind_gen_mw",
    "solar_gen_mw",
    "flow_dk1_de_mw",
    "de_da_price_eur_mwh",
    "se3_da_price_eur_mwh",
    "no2_da_price_eur_mwh",
    "shortwave_radiation",
    "cloud_cover",
    "temperature_2m",
    "wind_speed_100m",
    "wind_direction_100m",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "isoforest_score",
    "mahalanobis_24h",
    "bocpd_run_length_mean",
)
"""F8d — XAI-lean feature set (Track D.4). Derived from A16 ranking on F8c seed-13."""

ACTIVATION_ANOMALY_FEATURES = F8E_FEATURES + (
    "bocpd_run_length_mean",
    "bocpd_p_change",
    "bocpd_entropy",
    "zscore_24h",
    "zscore_96h",
    "hampel_flag_24h",
    "realized_vol_1h",
    "isoforest_score",
    "mahalanobis_24h",
)
"""Activation-volume forecaster features with causal anomaly scores over signed activation."""

F8WX72_FEATURES = F8B_FEATURES + (
    "wx_prev1_temperature_2m",
    "wx_prev2_temperature_2m",
    "wx_prev3_temperature_2m",
    "wx_prev1_wind_speed_10m",
    "wx_prev2_wind_speed_10m",
    "wx_prev3_wind_speed_10m",
    "wx_prev1_wind_direction_10m",
    "wx_prev2_wind_direction_10m",
    "wx_prev3_wind_direction_10m",
    "wx_prev1_shortwave_radiation",
    "wx_prev2_shortwave_radiation",
    "wx_prev3_shortwave_radiation",
    "wx_prev1_cloud_cover",
    "wx_prev2_cloud_cover",
    "wx_prev3_cloud_cover",
)
"""F8 weather-stack variant with +1/+2/+3 day Open-Meteo previous-run features."""

# ---------------------------------------------------------------------------
# F_CANONICAL — apples-to-apples maximal causally-honest feature panel.
# Used by `train.canonical` to compare every multivariate-capable F-zoo entry
# on identical inputs. Composition:
#   - F13_FEATURES  (F8e + 8 causal forecast-error features minus UMM/commodity nulls)
#   - F8c anomaly cols (BOCPD + iso-forest + Hampel + Mahalanobis + spread z-scores)
# All columns are causally honest: weather/gen actuals are 24h-lagged
# (trinity.DEFAULT_WEATHER_LAG_HOURS); BOCPD/anomaly are computed strictly
# on past windows; spreads use lagged-OK cross-zone DA. NEVER add lead-time
# columns here without proving causality first.
# ---------------------------------------------------------------------------
F_CANONICAL_ANOMALY_COLS = (
    "bocpd_run_length_mean",
    "bocpd_p_change",
    "bocpd_entropy",
    "zscore_24h",
    "zscore_96h",
    "hampel_flag_24h",
    "realized_vol_1h",
    "isoforest_score",
    "mahalanobis_24h",
    "dk1_de_spread_eur_mwh",
    "dk1_se3_spread_eur_mwh",
    "dk1_no2_spread_eur_mwh",
    "dk1_de_spread_zscore_24h",
)

F_CANONICAL_FEATURES = F13_FEATURES + F_CANONICAL_ANOMALY_COLS

# Feature groups for leave-one-group-out (LOGO) ablation. Drop one group at a
# time from F_CANONICAL_FEATURES to measure that group's contribution to
# pinball loss. The target column (imbalance price lag) is always kept.
CANONICAL_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "weather": (
        "wind_speed_100m", "wind_direction_100m", "temperature_2m",
        "shortwave_radiation", "cloud_cover",
    ),
    "cross_zone_da": (
        "de_da_price_eur_mwh", "se3_da_price_eur_mwh", "no2_da_price_eur_mwh",
    ),
    "cross_zone_spreads": (
        "dk1_de_spread_eur_mwh", "dk1_se3_spread_eur_mwh",
        "dk1_no2_spread_eur_mwh", "dk1_de_spread_zscore_24h",
    ),
    "generation": ("wind_gen_mw", "solar_gen_mw"),
    "mfrr_volume": ("mfrr_up_volume_mw", "mfrr_down_volume_mw"),
    "calendar": (
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_dk_holiday", "is_weekend",
    ),
    "anomaly_stat": (
        "bocpd_run_length_mean", "bocpd_p_change", "bocpd_entropy",
        "zscore_24h", "zscore_96h", "hampel_flag_24h", "realized_vol_1h",
        "isoforest_score", "mahalanobis_24h",
    ),
    "forecast_error": (
        "fe_offshore_wind_mw", "fe_onshore_wind_mw", "fe_solar_mw",
        "fe_total_renewable_mw", "fe_total_renewable_24h_mean",
        "fe_total_renewable_24h_std", "fe_total_renewable_6h_cumsum",
        "fe_total_renewable_zscore_24h",
    ),
    "load": ("load_actual_mw",),
    "da_price": ("da_price_dkk_mwh",),
}


def f_canonical_without(group: str) -> tuple[str, ...]:
    """Return F_CANONICAL_FEATURES minus ``CANONICAL_FEATURE_GROUPS[group]``.
    Raises KeyError if the group does not exist; logs a warning if the group
    has zero cols actually present in the canonical set (no-op).
    """
    drop = set(CANONICAL_FEATURE_GROUPS[group])
    kept = tuple(c for c in F_CANONICAL_FEATURES if c not in drop)
    return kept
"""Maximal apples-to-apples feature set for multivariate-capable forecasters.
Requires both ``dk1_panel_rich_v2_*.parquet`` AND ``anomaly_features.parquet``.
"""

# Routing: model_name -> "multivariate" | "univariate"
# - "multivariate" gets F_CANONICAL_FEATURES.
# - "univariate"   gets target-only (architectural constraint or pretrained
#                  univariate foundation model).
CANONICAL_MODEL_ROUTING: dict[str, str] = {
    # Multivariate (F_CANONICAL).
    "f1_lgbm": "multivariate",
    "f2_blr": "multivariate",
    "f5_np": "multivariate",
    "f6_anp": "multivariate",
    "f7": "multivariate",
    "f8": "multivariate",
    "f11": "multivariate",
    # Univariate by architecture (target-only).
    "ar1": "univariate",
    "f0": "univariate",
    "f3_lite": "univariate",   # DeepARLite has no covariate channel (appendix-only; see ADR-0006).
    # Pretrained foundation univariate models (cannot accept covariates).
    "f9": "univariate",
    "f10": "univariate",
}


@dataclass
class WindowStats:
    """Per-feature mean/std from the training window. Frozen at train time and
    re-applied on val/test (no test-set leakage). Also carries scalar target
    mean/std so the trainer can de-normalise predictions for reporting."""

    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]
    target_name: str = TARGET_COL
    target_mean: float = 0.0
    target_std: float = 1.0

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.where(self.std == 0.0, 1.0, self.std)

    def normalise_target(self, y: np.ndarray) -> np.ndarray:
        s = self.target_std if self.target_std > 0 else 1.0
        return (y - self.target_mean) / s

    def denormalise_target(self, y: np.ndarray) -> np.ndarray:
        s = self.target_std if self.target_std > 0 else 1.0
        return y * s + self.target_mean


def _read_panel(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    # The panel may carry stray nulls at boundaries; drop them.
    return df.drop_nulls()


def make_windows(
    panel_path: Path,
    *,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    multivariate: bool = False,
    feature_names: tuple[str, ...] | None = None,
    target: TargetKind = "price",
    target_column: str | None = None,
    stats: WindowStats | None = None,
    anomaly_panel_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, WindowStats]:
    """Build (X, Y, stats) where X has shape (N, seq_len, F) and Y (N, horizon).

    Parameters
    ----------
    panel_path
        Parquet to use for inputs + target. ``data/processed/dk1_panel.parquet``
        for F7/F8a; ``data/processed/dk1_panel_rich.parquet`` for F8b/c/d/F12.
    multivariate
        Legacy switch — selects ``F8_FEATURES`` (= F8a) when ``feature_names`` is
        not given. Ignored if ``feature_names`` is explicitly passed.
    feature_names
        Explicit feature tuple. Use ``F8B_FEATURES`` / ``F8C_FEATURES`` /
        the lean F8d set as produced by Track D.
    anomaly_panel_path
        Optional path to ``data/processed/anomaly_features.parquet`` to be
        joined onto ``panel_path`` on ``timestamp_utc`` before windowing. Used
        for F8c/F8d which consume BOCPD + iso-forest + spread features.
    target
        ``price`` uses ``imbalance_price_dkk_mwh_15min``; ``activation_volume``
        uses signed activation MWh (up positive, down negative); and
        ``activation_direction`` uses class labels ``1`` / ``0`` / ``-1``.
    """
    df = _read_panel(panel_path)
    if anomaly_panel_path is not None and Path(anomaly_panel_path).exists():
        af = _read_panel(Path(anomaly_panel_path))
        df = df.join(af, on="timestamp_utc", how="left").drop_nulls()
    if feature_names is None:
        feature_names = F8_FEATURES if multivariate else (TARGET_COL,)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise KeyError(
            f"Panel missing required features {missing}. "
            f"Available: {df.columns}. Tip: rich panel + anomaly_panel_path needed for F8b+."
        )
    arr = df.select(list(feature_names)).to_numpy().astype(np.float64)
    target_values, target_name = make_target(df, target=target, target_column=target_column)

    if stats is None:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0) + 1e-8
        stats = WindowStats(
            mean=mean,
            std=std,
            feature_names=feature_names,
            target_name=target_name,
            target_mean=float(target_values.mean()),
            target_std=float(target_values.std() + 1e-8),
        )
    arr_norm = stats.normalise(arr)
    target_norm = stats.normalise_target(target_values)

    n_windows = arr.shape[0] - seq_len - horizon + 1
    if n_windows <= 0:
        raise ValueError(
            f"panel too short for seq_len={seq_len} horizon={horizon}: rows={arr.shape[0]}"
        )
    X = np.empty((n_windows, seq_len, arr.shape[1]), dtype=np.float32)
    Y = np.empty((n_windows, horizon), dtype=np.float32)
    for i in range(n_windows):
        X[i] = arr_norm[i : i + seq_len]
        Y[i] = target_norm[i + seq_len : i + seq_len + horizon]
    return X, Y, stats


def make_target(
    df: pl.DataFrame,
    *,
    target: TargetKind = "price",
    target_column: str | None = None,
) -> tuple[np.ndarray, str]:
    """Return a 1D target array and logical name for price/activation tasks."""
    if target_column is not None:
        if target_column not in df.columns:
            raise KeyError(f"Panel missing target_column={target_column!r}")
        return df.select(target_column).to_numpy().astype(np.float64).ravel(), target_column
    if target == "price":
        return df.select(TARGET_COL).to_numpy().astype(np.float64).ravel(), TARGET_COL
    if target in {"activation_volume", "activation_direction"}:
        for col in ("mfrr_up_volume_mw", "mfrr_down_volume_mw"):
            if col not in df.columns:
                raise KeyError(f"Panel missing required activation column {col!r}")
        up = df.select("mfrr_up_volume_mw").to_numpy().astype(np.float64).ravel()
        down = df.select("mfrr_down_volume_mw").to_numpy().astype(np.float64).ravel()
        if target == "activation_volume":
            return (up - down) * 0.25, "signed_activation_volume_mwh"
        labels = np.where(up > down, 1.0, np.where(down > up, -1.0, 0.0))
        return labels.astype(np.float64), "activation_direction_label"
    raise ValueError(f"unknown target={target!r}")


class QuantilePanelDataset(Dataset):
    """Wraps (X, Y) numpy arrays as torch tensors for the DataLoader."""

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


__all__ = [
    "F8B_FEATURES",
    "F8C_FEATURES",
    "F8E_FEATURES",
    "F8D_FEATURES",
    "F13_FEATURES",
    "F_CANONICAL_FEATURES",
    "F_CANONICAL_ANOMALY_COLS",
    "CANONICAL_MODEL_ROUTING",
    "CANONICAL_FEATURE_GROUPS",
    "f_canonical_without",
    "ACTIVATION_ANOMALY_FEATURES",
    "F8WX72_FEATURES",
    "F8_FEATURES",
    "HORIZON",
    "QUANTILES",
    "QuantilePanelDataset",
    "SEQ_LEN",
    "TARGET_COL",
    "make_target",
    "WindowStats",
    "make_windows",
]
