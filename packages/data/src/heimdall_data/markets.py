"""Fuel + carbon market price loaders for the DK1 panel.

Per Plan v2 Track A.4. Loads pre-fetched daily settlement CSVs/parquets from
``data/external/markets/`` and forward-fills onto the panel's 15-min grid.

The columns are:

  - gas_ttf_eur_mwh : TTF natural gas daily settle (EUR / MWh thermal)
  - eu_ets_eur_t    : EU-ETS carbon front-month daily settle (EUR / tonne CO₂)
  - coal_api2_usd_t : API2 coal CIF ARA daily settle (USD / tonne)
  - brent_usd_bbl   : Brent crude oil daily settle (USD / barrel)

These series are slow-moving (daily) and globally available. We do NOT fetch
them at runtime — there is a separate one-shot script ``tools/fetch_markets.py``
that the user runs to populate ``data/external/markets/``.

If a series is absent, that column is silently emitted as null. Downstream
forecasters MUST tolerate missing-by-construction.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

MARKETS_ROOT = Path(__file__).resolve().parents[4] / "data" / "external" / "markets"

MARKET_SERIES: dict[str, str] = {
    "gas_ttf_eur_mwh": "ttf_gas.parquet",
    "eu_ets_eur_t": "eu_ets.parquet",
    "coal_api2_usd_t": "api2_coal.parquet",
    "brent_usd_bbl": "brent_oil.parquet",
}
"""Logical column name → file under data/external/markets/.

Each file must have schema (date: Date, value: Float64). The loader rolls them
into the 15-min panel grid via forward-fill from the daily settle (settle is
known by ~17:30 UTC of the same day; for safety we forward-fill from 18:00 UTC
of day D into all 15-min slots of day D+1, eliminating intraday-leak)."""


def _load_one_series(name: str, filename: str, root: Path) -> pl.DataFrame | None:
    p = root / filename
    if not p.exists():
        return None
    df = pl.read_parquet(p)
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{p}: expected columns (date, value); got {df.columns}")
    # Convert date → utc 18:00 (publication time, end of trading day in Europe).
    # Forward-fill thereafter into the next trading day.
    df = df.with_columns(
        pl.col("date")
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
        .dt.offset_by("18h")
        .alias("timestamp_utc"),
        pl.col("value").cast(pl.Float64).alias(name),
    ).select(["timestamp_utc", name])
    return df.sort("timestamp_utc")


def load_markets(
    start_utc, end_utc, *, root: Path | None = None
) -> pl.DataFrame:
    """Return a wide hourly frame of fuel+carbon columns over [start, end).

    Frame skeleton is an hourly grid; values come from daily settles
    forward-filled at the 18:00 UTC publication boundary. If a column file is
    missing, that column is all-null.
    """
    r = root or MARKETS_ROOT
    r.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    rng = pd.date_range(start=start_utc, end=end_utc, freq="1h", tz="UTC", inclusive="left")
    base = pl.DataFrame(
        {
            "timestamp_utc": pl.Series(
                rng.to_pydatetime().tolist(),
                dtype=pl.Datetime("us", time_zone="UTC"),
            )
        }
    )

    for col, fname in MARKET_SERIES.items():
        df = _load_one_series(col, fname, r)
        if df is None:
            base = base.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
            continue
        base = base.join(df, on="timestamp_utc", how="left").sort("timestamp_utc")
        base = base.with_columns(pl.col(col).forward_fill())

    return base


__all__ = ["MARKETS_ROOT", "MARKET_SERIES", "load_markets"]
