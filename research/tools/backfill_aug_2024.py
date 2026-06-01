"""One-shot backfill for the missing 2024-08 DK1 monthly panel.

The canonical fetch (`tools/fetch_dk1_history.py`) needs `ENTSOE_API_TOKEN`
for `load_actual_mw`. This script substitutes ENTSO-E with the public DK1
load parquet from the sister project `the-traders-trinity` (already cloned
to `/tmp/ttt/`), and pulls every other column from Energinet's free API.

Writes `data/processed/dk1_panel_20240801_20240901.parquet` matching the
schema of the surrounding monthlies, then re-builds the canonical
train/val/test splits.

Sources:
  - Energinet RegulatingBalancePowerdata (DK1, free) -> imbalance prices, mFRR
  - Energinet Elspotprices (DK1, free) -> da_price_dkk_mwh
  - the-traders-trinity actual_load.parquet (DK1 hourly) -> load_actual_mw
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import polars as pl

from heimdall_data.energinet import EnerginetClient

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/processed/dk1_panel_20240801_20240901.parquet"
TTT_LOAD = Path("/tmp/ttt/data/raw/entsoe/actual_load.parquet")

START = datetime(2024, 8, 1, tzinfo=timezone.utc)
END = datetime(2024, 9, 1, tzinfo=timezone.utc)
ISO_START, ISO_END = "2024-08-01T00:00", "2024-09-01T00:00"


def _grid() -> pl.DataFrame:
    rng = pd.date_range(START, END, freq="15min", tz="UTC", inclusive="left")
    return pl.DataFrame({
        "timestamp_utc": pl.Series(
            "timestamp_utc",
            rng.to_pydatetime().tolist(),
        ).cast(pl.Datetime("us", time_zone="UTC"))
    })


def _ffill_join(grid: pl.DataFrame, src: pd.DataFrame, value_col: str, out_col: str) -> pl.DataFrame:
    if src.empty:
        return grid.with_columns(pl.lit(None, dtype=pl.Float64).alias(out_col))
    src = src[["timestamp_utc", value_col]].dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    src["timestamp_utc"] = pd.to_datetime(src["timestamp_utc"], utc=True)
    gpd = grid.to_pandas().sort_values("timestamp_utc")
    gpd["timestamp_utc"] = pd.to_datetime(gpd["timestamp_utc"], utc=True).astype("datetime64[ns, UTC]")
    src["timestamp_utc"] = src["timestamp_utc"].astype("datetime64[ns, UTC]")
    merged = pd.merge_asof(gpd, src, on="timestamp_utc", direction="backward",
                           tolerance=pd.Timedelta("1h"))
    merged = merged.rename(columns={value_col: out_col})
    return pl.from_pandas(merged[["timestamp_utc", out_col]]).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("us", time_zone="UTC"))
    )


def main() -> int:
    print("[backfill] pulling Energinet RegulatingBalancePowerdata 2024-08 DK1 ...")
    cli = EnerginetClient()
    rb = cli.regulating_balance(ISO_START, ISO_END, "DK1")
    print(f"  rows={len(rb)}")
    print("[backfill] pulling Energinet Elspotprices 2024-08 DK1 ...")
    sp = cli.elspot_prices(ISO_START, ISO_END, "DK1")
    print(f"  rows={len(sp)}")

    rb_df = pd.DataFrame(rb)
    rb_df["timestamp_utc"] = pd.to_datetime(rb_df["HourUTC"], utc=True)
    sp_df = pd.DataFrame(sp)
    sp_df["timestamp_utc"] = pd.to_datetime(sp_df["HourUTC"], utc=True)

    # Map Energinet column names -> canonical schema
    rb_df["imbalance_price_dkk_mwh"] = pd.to_numeric(rb_df.get("ImbalancePriceDKK"), errors="coerce")
    rb_df["imbalance_price_eur_mwh"] = pd.to_numeric(rb_df.get("ImbalancePriceEUR"), errors="coerce")
    rb_df["mfrr_up_volume_mw"] = pd.to_numeric(rb_df.get("mFRRUpActBal"), errors="coerce")
    rb_df["mfrr_down_volume_mw"] = pd.to_numeric(rb_df.get("mFRRDownActBal"), errors="coerce")
    sp_df["da_price_dkk_mwh"] = pd.to_numeric(sp_df.get("SpotPriceDKK"), errors="coerce")

    # Load from ttt actual_load.parquet (DK1 hourly)
    load = pl.read_parquet(TTT_LOAD).filter(pl.col("zone") == "DK1").sort("utc_timestamp")
    load_pd = load.select(["utc_timestamp", "load_mw"]).rename({"utc_timestamp": "timestamp_utc"}).to_pandas()
    load_pd = load_pd[(load_pd["timestamp_utc"] >= START) & (load_pd["timestamp_utc"] < END)]
    print(f"  ttt DK1 load rows in window={len(load_pd)}")

    grid = _grid()
    panel = grid
    panel = panel.join(
        _ffill_join(grid, sp_df, "da_price_dkk_mwh", "da_price_dkk_mwh"),
        on="timestamp_utc", how="left",
    )
    panel = panel.join(
        _ffill_join(grid, load_pd, "load_mw", "load_actual_mw"),
        on="timestamp_utc", how="left",
    )
    panel = panel.join(
        _ffill_join(grid, rb_df, "imbalance_price_dkk_mwh", "imbalance_price_dkk_mwh"),
        on="timestamp_utc", how="left",
    )
    panel = panel.join(
        _ffill_join(grid, rb_df, "mfrr_up_volume_mw", "mfrr_up_volume_mw"),
        on="timestamp_utc", how="left",
    )
    panel = panel.join(
        _ffill_join(grid, rb_df, "mfrr_down_volume_mw", "mfrr_down_volume_mw"),
        on="timestamp_utc", how="left",
    )
    panel = panel.join(
        _ffill_join(grid, rb_df, "imbalance_price_eur_mwh", "imbalance_price_eur_mwh"),
        on="timestamp_utc", how="left",
    )
    # Pre-2025-03-04: 15-min A85 unavailable, fall back to hourly imbalance.
    panel = panel.with_columns(
        pl.col("imbalance_price_dkk_mwh").alias("imbalance_price_dkk_mwh_15min")
    )
    # Match schema order with neighbouring monthlies.
    panel = panel.select([
        "timestamp_utc",
        "da_price_dkk_mwh",
        "load_actual_mw",
        "imbalance_price_dkk_mwh",
        "mfrr_up_volume_mw",
        "mfrr_down_volume_mw",
        "imbalance_price_eur_mwh",
        "imbalance_price_dkk_mwh_15min",
    ])
    null_counts = {c: int(panel[c].null_count()) for c in panel.columns}
    print("[backfill] null counts:", null_counts)
    panel.write_parquet(OUT)
    print(f"[backfill] wrote {OUT} rows={len(panel)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
