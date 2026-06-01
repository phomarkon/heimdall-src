"""Build CAUSAL anomaly features from DA forecast vs realised generation.

Why this is causal: the DA forecast is published the day BEFORE delivery
(D-1 at 12:00 CET). At any 15-min quarter t in delivery day D, both the
DA forecast and the realised generation up to t-1 are known. We compute:

    forecast_error_wind_offshore_mw[t]   = realised_offshore_wind[t] - forecast_offshore_wind[t]
    forecast_error_wind_onshore_mw[t]    = ...
    forecast_error_solar_mw[t]           = ...
    forecast_error_total_renewable_mw[t] = sum of the three above

Plus rolling statistics (24h z-score, 6h cumulative) to expose the
*regime* of forecast deviation, not just the instantaneous value.

These features are then joined onto data/processed/dk1_panel_rich.parquet
to produce dk1_panel_rich_v2.parquet, which F13 trains on.

Output:
  data/processed/dk1_panel_rich_v2.parquet   (rich + forecast errors)
  data/processed/dk1_panel_rich_v2_train.parquet  (split by EAM break)
  data/processed/dk1_panel_rich_v2_val.parquet
  data/processed/dk1_panel_rich_v2_test.parquet
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data/processed"
TRAIN_END = datetime(2025, 3, 1, tzinfo=timezone.utc)
VAL_START = datetime(2025, 3, 4, tzinfo=timezone.utc)
VAL_END = datetime(2025, 5, 1, tzinfo=timezone.utc)
TEST_START = VAL_END
TEST_END = datetime(2026, 5, 1, tzinfo=timezone.utc)


def main() -> int:
    print("[fe] loading features_v2 + rich panels ...")
    fv2 = pl.read_parquet(PROCESSED / "dk1_panel_features_v2.parquet").sort("timestamp_utc")
    rich = pl.read_parquet(PROCESSED / "dk1_panel_rich.parquet").sort("timestamp_utc")

    # The features_v2 panel uses MWh-per-quarter (sub-MW) actuals; multiply by 4
    # to get instantaneous MW values matching the forecast units (MW).
    fv2 = fv2.with_columns([
        ((pl.col("OffshoreWindLt100MW_MWh").fill_null(0.0) +
          pl.col("OffshoreWindGe100MW_MWh").fill_null(0.0)) * 4).alias("actual_offshore_wind_mw"),
        ((pl.col("OnshoreWindLt50kW_MWh").fill_null(0.0) +
          pl.col("OnshoreWindGe50kW_MWh").fill_null(0.0)) * 4).alias("actual_onshore_wind_mw"),
        ((pl.col("SolarPowerLt10kW_MWh").fill_null(0.0) +
          pl.col("SolarPowerGe10Lt40kW_MWh").fill_null(0.0) +
          pl.col("SolarPowerGe40kW_MWh").fill_null(0.0)) * 4).alias("actual_solar_mw"),
        pl.col("GrossConsumptionMWh").fill_null(0.0).mul(4).alias("actual_load_total_mw"),
    ])

    # Forecast errors (realised - forecast). Negative -> over-forecast (less
    # renewable than expected -> imbalance pushes price UP, regulation up).
    fv2 = fv2.with_columns([
        (pl.col("actual_offshore_wind_mw") - pl.col("forecast_da_mw_offshore_wind").fill_null(0.0))
            .alias("fe_offshore_wind_mw"),
        (pl.col("actual_onshore_wind_mw") - pl.col("forecast_da_mw_onshore_wind").fill_null(0.0))
            .alias("fe_onshore_wind_mw"),
        (pl.col("actual_solar_mw") - pl.col("forecast_da_mw_solar").fill_null(0.0))
            .alias("fe_solar_mw"),
    ])
    fv2 = fv2.with_columns([
        (pl.col("fe_offshore_wind_mw") + pl.col("fe_onshore_wind_mw") + pl.col("fe_solar_mw"))
            .alias("fe_total_renewable_mw"),
    ])

    # 24h rolling stats on the total forecast error (96 quarters = 1 day).
    fv2 = fv2.with_columns([
        pl.col("fe_total_renewable_mw").rolling_mean(window_size=96, min_periods=24)
            .alias("fe_total_renewable_24h_mean"),
        pl.col("fe_total_renewable_mw").rolling_std(window_size=96, min_periods=24)
            .alias("fe_total_renewable_24h_std"),
        # 6h cumulative — magnitude of recent net deviation
        pl.col("fe_total_renewable_mw").rolling_sum(window_size=24, min_periods=4)
            .alias("fe_total_renewable_6h_cumsum"),
    ])
    # Z-score: how anomalous is the CURRENT forecast error?
    fv2 = fv2.with_columns([
        ((pl.col("fe_total_renewable_mw") - pl.col("fe_total_renewable_24h_mean"))
         / (pl.col("fe_total_renewable_24h_std").fill_null(1.0).clip(lower_bound=1.0)))
            .alias("fe_total_renewable_zscore_24h"),
    ])

    keep = [
        "timestamp_utc",
        "fe_offshore_wind_mw", "fe_onshore_wind_mw", "fe_solar_mw",
        "fe_total_renewable_mw",
        "fe_total_renewable_24h_mean", "fe_total_renewable_24h_std",
        "fe_total_renewable_6h_cumsum", "fe_total_renewable_zscore_24h",
    ]
    fe = fv2.select(keep).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("us", time_zone="UTC"))
    )
    rich = rich.with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("us", time_zone="UTC"))
    )
    print(f"  forecast-error panel: {fe.shape}")

    rich_v2 = rich.join(fe, on="timestamp_utc", how="left").sort("timestamp_utc")
    print(f"[fe] rich_v2 shape: {rich_v2.shape} (was {rich.shape})")

    rich_v2.write_parquet(PROCESSED / "dk1_panel_rich_v2.parquet")

    # Splits matching the rest of the zoo (frozen 2025-03-04 break).
    for name, lo, hi in [
        ("train", None, TRAIN_END),
        ("val", VAL_START, VAL_END),
        ("test", TEST_START, TEST_END),
    ]:
        df = rich_v2
        if lo is not None:
            df = df.filter(pl.col("timestamp_utc") >= lo)
        if hi is not None:
            df = df.filter(pl.col("timestamp_utc") < hi)
        df = df.filter(pl.col("imbalance_price_dkk_mwh_15min").is_not_null())
        out = PROCESSED / f"dk1_panel_rich_v2_{name}.parquet"
        df.write_parquet(out)
        print(f"  wrote {out}: shape={df.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
