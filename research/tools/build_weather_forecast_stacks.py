"""Build weather forecast-stack rich panels for ablations.

Creates:
  - data/processed/dk1_panel_rich_wx72.parquet
  - data/processed/dk1_panel_rich_wx72_{train,val,test}.parquet

The weather stack comes from Open-Meteo Previous Runs API using fixed lead-time
offsets (previous_day1/2/3), which are causally valid for backtests.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pandas as pd

from heimdall_data.open_meteo import WeatherLocation, get_cached_previous_runs_forecast

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data/processed"
PRE_POST = datetime(2025, 3, 4, 0, 0, tzinfo=UTC)
TEST_START = datetime(2025, 5, 1, 0, 0, tzinfo=UTC)
TEST_END = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)


def _split_and_write(df: pl.DataFrame, stem: str) -> None:
    train = df.filter(pl.col("timestamp_utc") < PRE_POST)
    val = df.filter((pl.col("timestamp_utc") >= PRE_POST) & (pl.col("timestamp_utc") < TEST_START))
    test = df.filter((pl.col("timestamp_utc") >= TEST_START) & (pl.col("timestamp_utc") < TEST_END))
    train.write_parquet(PROCESSED / f"{stem}_train.parquet")
    val.write_parquet(PROCESSED / f"{stem}_val.parquet")
    test.write_parquet(PROCESSED / f"{stem}_test.parquet")
    print(f"{stem}: train={train.height:,} val={val.height:,} test={test.height:,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(PROCESSED / "dk1_panel_rich.parquet"))
    parser.add_argument("--out", default=str(PROCESSED / "dk1_panel_rich_wx72.parquet"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--lat", type=float, default=56.26, help="DK1 centroid latitude")
    parser.add_argument("--lon", type=float, default=9.50, help="DK1 centroid longitude")
    parser.add_argument("--start", type=str, default=None, help="Override start UTC ISO timestamp")
    parser.add_argument("--end", type=str, default=None, help="Override end UTC ISO timestamp")
    args = parser.parse_args(argv)

    base_path = Path(args.base)
    out_path = Path(args.out)
    if not base_path.exists():
        raise FileNotFoundError(f"missing base panel: {base_path}")
    base = pl.read_parquet(base_path).sort("timestamp_utc")
    start = base["timestamp_utc"].min()
    end = base["timestamp_utc"].max()
    if start is None or end is None:
        raise ValueError("base panel is empty")
    if args.start:
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(UTC)
    if args.end:
        end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(UTC)

    loc = WeatherLocation(zone="DK1", latitude=args.lat, longitude=args.lon)
    vars_ = ("temperature", "wind_speed", "wind_direction", "solar_radiation", "cloud_cover")
    # Chunk by month to avoid oversized single calls to the API.
    chunks = []
    cur = start
    while cur <= end:
        nxt = min(cur + pd.DateOffset(days=31), end)
        cached = get_cached_previous_runs_forecast(
            loc,
            variables=vars_,
            start=cur,
            end=nxt,
            lead_days=(1, 2, 3),
            refresh=args.refresh,
        )
        chunks.append(cached.frame)
        print(f"fetched chunk: {cur} -> {nxt} rows={len(cached.frame):,}")
        cur = nxt
    wx_pdf = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    wx = pl.from_pandas(wx_pdf).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("us", time_zone="UTC"))
    )
    wx_cols = [c for c in wx.columns if c not in {"timestamp_utc", "zone"}]
    merged = base.join(wx.select(["timestamp_utc", *wx_cols]), on="timestamp_utc", how="left")
    merged = merged.with_columns(
        [
            pl.col(c).fill_null(strategy="forward").fill_null(0.0)
            for c in wx_cols
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path)
    print(f"wrote {out_path} rows={merged.height:,} cols={len(merged.columns)}")
    _split_and_write(merged, out_path.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
