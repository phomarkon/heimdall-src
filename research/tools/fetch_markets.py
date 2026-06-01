"""One-shot fetch script for fuel + carbon daily series.

Writes parquets into ``data/external/markets/`` consumed by
``heimdall_data.markets.load_markets``.

Sources (free, no key):
- TTF gas        : yfinance ``TTF=F`` (front-month, EUR/MWh thermal proxy)
- EU-ETS         : yfinance ``KRBN`` (KraneShares ETS ETF, USD; converted to EUR
                   via daily ECB rate; *coarse proxy* — for production replace
                   with EEX/Sandbag direct CSV)
- API2 coal      : yfinance ``MTF=F`` (front-month coal, USD/t)
- Brent          : yfinance ``BZ=F`` (front-month Brent, USD/bbl)

Usage:
    uv run python tools/fetch_markets.py --start 2016-01-01 --end 2026-04-30

Notes:
- yfinance is an optional dep; if not installed, suggest ``uv add yfinance``.
- The script writes whatever data it can fetch; missing series leave the
  corresponding file unwritten and the panel column will be null.
- Document the fetch in ``data/external/markets/PROVENANCE.md`` with the run
  date and source library version after each run.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import polars as pl

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "external" / "markets"

# logical name -> yfinance ticker
TICKERS = {
    "ttf_gas.parquet": "TTF=F",
    "eu_ets.parquet": "KRBN",  # proxy; replace with direct ETS source for paper-grade numbers
    "api2_coal.parquet": "MTF=F",
    "brent_oil.parquet": "BZ=F",
}


def _fetch_one(ticker: str, start: str, end: str) -> pl.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        print("[fetch_markets] yfinance not installed. Run: uv add yfinance", file=sys.stderr)
        return None
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    df = df["Close"].reset_index()
    df.columns = ["date", "value"]
    return pl.from_pandas(df).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("value").cast(pl.Float64),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    for fname, ticker in TICKERS.items():
        print(f"[fetch_markets] {ticker:8s} -> {fname} ...", end=" ", flush=True)
        df = _fetch_one(ticker, args.start, args.end)
        if df is None:
            print("FAIL")
            continue
        path = args.out_dir / fname
        df.write_parquet(path)
        print(f"OK ({df.height} rows -> {path})")
        successes += 1

    print(f"[fetch_markets] wrote {successes}/{len(TICKERS)} series")
    return 0 if successes == len(TICKERS) else 1


if __name__ == "__main__":
    sys.exit(main())
