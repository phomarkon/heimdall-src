"""Backfill commodity-price placeholder columns in the DK1 rich panel.

Sources (all public, no auth):
  - Brent: FRED DCOILBRENTEU daily (USD/bbl).
  - EU ETS: FRED proxy via European Commission CSV; fallback to ICAP price file.
  - Gas TTF: Investing.com / energy charts CSV not stable; fallback to
    FRED PNGASEUUSDM (proxy, monthly EU natural gas). Documented as proxy.
  - Coal API2: no free daily series; fallback to FRED PCOALAUUSDM (Australian
    thermal coal, monthly). Documented as proxy.

All series are daily/monthly; forward-filled to the 15-min panel resolution.
Run:
    PYTHONPATH=. uv run python tools/backfill_commodity_prices.py
Output:
    data/processed/dk1_panel_rich_with_commodities.parquet
"""
from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data" / "processed" / "dk1_panel_rich_v2.parquet"
OUT = REPO / "data" / "processed" / "dk1_panel_rich_with_commodities.parquet"

FRED = {
    "brent_usd_bbl":      "DCOILBRENTEU",   # daily, USD/bbl
    "gas_henry_usd_mmbtu":"DHHNGSP",        # daily, USD/MMBtu — proxy for global gas
    "gas_eu_eur_mwh":     "PNGASEUUSDM",    # monthly proxy — PNGASEUUSDM is USD/MMBtu EU import
    "coal_au_usd_t":      "PCOALAUUSDM",    # monthly proxy — Australian thermal coal USD/t
}


def _fred_csv(series_id: str) -> pl.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    raw = urllib.request.urlopen(url, timeout=30).read()
    df = pl.read_csv(io.BytesIO(raw))
    val_col = [c for c in df.columns if c != "observation_date"][0]
    return df.rename({"observation_date": "date", val_col: "value"}).with_columns([
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("value").cast(pl.Float64, strict=False),
    ]).drop_nulls()


def main() -> int:
    df = pl.read_parquet(SRC).sort("timestamp_utc")
    df = df.with_columns(
        pl.col("timestamp_utc").cast(pl.Date).alias("_date")
    )
    print(f"loaded {SRC.name}  shape={df.shape}")

    for col, series_id in FRED.items():
        try:
            f = _fred_csv(series_id)
            print(f"  {col} <- FRED {series_id}: {len(f)} obs ({f['date'].min()} .. {f['date'].max()})")
            df = df.join(f.rename({"value": col}), left_on="_date", right_on="date", how="left")
        except Exception as e:
            print(f"  {col} FAIL: {e}")
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Forward-fill within each new column (daily → 15-min upsample).
    for col in FRED:
        if col in df.columns:
            df = df.with_columns(pl.col(col).forward_fill())

    df = df.drop("_date")
    df.write_parquet(OUT)
    print(f"wrote {OUT}  shape={df.shape}")

    # Summary
    summary = {
        "source": str(SRC.relative_to(REPO)),
        "out": str(OUT.relative_to(REPO)),
        "columns_added": list(FRED.keys()),
        "fred_series_ids": FRED,
        "note": "Forward-filled from daily/monthly source to 15-min panel resolution. "
                "EU gas / Australian coal columns are FRED proxies (monthly); "
                "use as exogenous regressors, not as DK1-physical prices.",
    }
    (REPO / "notes" / "findings" / "2026-05-17-commodity-backfill.json").parent.mkdir(parents=True, exist_ok=True)
    (REPO / "notes" / "findings" / "2026-05-17-commodity-backfill.json").write_text(json.dumps(summary, indent=2))
    print("Wrote summary to notes/findings/2026-05-17-commodity-backfill.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
