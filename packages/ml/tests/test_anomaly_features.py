"""Tests for heimdall_ml.features.anomaly.

Smoke + causality + sklearn-train-only-fit checks. Constructs synthetic
panels so tests are network-free and fast (<5s).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from heimdall_ml.features.anomaly import (
    PRE_POST_BREAK_UTC,
    _hampel_flag,
    _realized_vol,
    _rolling_z,
    build_anomaly_features,
)


def _synthetic_panel(n: int = 4 * 24 * 30, seed: int = 13) -> pl.DataFrame:
    """A 30-day 15-min synthetic DK1-ish panel for tests."""
    rng = np.random.default_rng(seed)
    import pandas as pd

    ts = pd.date_range("2024-12-01 00:00", periods=n, freq="15min", tz="UTC")
    base = 300 + 50 * np.sin(np.arange(n) * 2 * np.pi / 96)  # daily seasonality
    noise = rng.normal(0, 20, n)
    price = base + noise

    return pl.DataFrame(
        {
            "timestamp_utc": pl.Series(
                ts.to_pydatetime().tolist(),
                dtype=pl.Datetime("us", time_zone="UTC"),
            ),
            "imbalance_price_dkk_mwh": price,
            "load_actual_mw": 2000 + 500 * np.cos(np.arange(n) * 2 * np.pi / 96) + rng.normal(0, 100, n),
            "wind_gen_mw": np.maximum(0, 1500 + rng.normal(0, 600, n)),
            "solar_gen_mw": np.maximum(0, 800 * np.sin(np.arange(n) * 2 * np.pi / 96) ** 2),
            "wind_speed_100m": np.abs(rng.normal(10, 4, n)),
            "de_da_price_eur_mwh": (price / 7.46) + rng.normal(0, 5, n),
            "se3_da_price_eur_mwh": (price / 7.46) + rng.normal(0, 3, n),
            "no2_da_price_eur_mwh": (price / 7.46) + rng.normal(0, 8, n),
        }
    )


def test_rolling_z_is_causal() -> None:
    """Output[t] must depend only on input[:t+1]."""
    rng = np.random.default_rng(0)
    full = rng.normal(0, 1, 1000)
    z_full = _rolling_z(full, window=50)
    # Truncate at midpoint and recompute — first half must match.
    half = full[:500]
    z_half = _rolling_z(half, window=50)
    np.testing.assert_allclose(z_full[:500], z_half, equal_nan=True)


def test_hampel_flag_is_binary() -> None:
    # Need a non-constant window so MAD > 0 (Hampel correctly abstains on degenerate input).
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 100), np.array([50.0])])
    out = _hampel_flag(x, window=50)
    assert ((out == 0.0) | (out == 1.0)).all()
    assert out[-1] == 1.0  # outlier flagged


def test_realized_vol_nonneg() -> None:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 500))
    out = _realized_vol(x, window=4)
    assert ((out >= 0) | np.isnan(out)).all()


def test_build_anomaly_features_schema_and_speed(tmp_path) -> None:
    import time

    panel = _synthetic_panel()
    t0 = time.time()
    out = build_anomaly_features(rich_panel=panel, output_path=tmp_path / "af.parquet", seed=13)
    dur = time.time() - t0
    assert dur < 15.0, f"anomaly build too slow on small synthetic: {dur:.1f}s"
    for col in (
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
        "dk1_de_spread_zscore_24h",
    ):
        assert col in out.columns, f"missing {col}"
    assert out.height == panel.height


def test_isoforest_fit_only_on_train_fold(tmp_path) -> None:
    """Sanity: the median imputation values must come from the train portion only.

    We construct a panel where the post-break window contains very different values;
    if iso-forest were fit on the whole panel it would treat post-break as "normal"
    and the score there would be inflated. With train-only fit, post-break should
    score as anomalous on average.
    """
    n = 4 * 24 * 60  # 60 days, with PRE_POST_BREAK_UTC midway
    import pandas as pd

    ts = pd.date_range(
        PRE_POST_BREAK_UTC - pd.Timedelta(days=30),
        periods=n,
        freq="15min",
        tz="UTC",
    )
    rng = np.random.default_rng(0)
    price = np.where(
        np.array(ts.to_pydatetime()) < PRE_POST_BREAK_UTC,
        rng.normal(300, 20, n),
        rng.normal(900, 100, n),  # very different post-break regime
    )
    panel = pl.DataFrame(
        {
            "timestamp_utc": pl.Series(
                ts.to_pydatetime().tolist(),
                dtype=pl.Datetime("us", time_zone="UTC"),
            ),
            "imbalance_price_dkk_mwh": price,
            "load_actual_mw": rng.normal(2000, 200, n),
            "wind_gen_mw": rng.normal(1500, 400, n),
            "solar_gen_mw": np.maximum(0, rng.normal(500, 200, n)),
            "wind_speed_100m": np.abs(rng.normal(10, 3, n)),
        }
    )
    out = build_anomaly_features(rich_panel=panel, output_path=tmp_path / "af.parquet", seed=13)
    train_mask = (out["timestamp_utc"] < PRE_POST_BREAK_UTC).to_numpy()
    iso_train = out["isoforest_score"].to_numpy()[train_mask]
    iso_post = out["isoforest_score"].to_numpy()[~train_mask]
    assert iso_train.mean() > iso_post.mean(), (
        "Train fold should score as more normal than post-break under "
        f"train-only fit; got train_mean={iso_train.mean():.3f}, "
        f"post_mean={iso_post.mean():.3f}"
    )
