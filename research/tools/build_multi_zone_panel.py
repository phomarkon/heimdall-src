"""Build the multi-zone DA panel for F12 pretraining (Plan v2 Track A.8).

Long-form schema: (utc_timestamp, zone, price_eur_mwh).

Source: Trinity prices.parquet at data/external/trinity/prices.parquet,
filtered to F12's training zones (DK1, DE, SE3, NO2) and a configurable
time window. The F12 pretrain set is by construction *temporally outside* the
DK1 test window (it ends at 2025-03-03 — train cutoff per proposal §5.7).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from heimdall_data.trinity import load_trinity_prices

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "data" / "processed" / "multi_zone_da_panel.parquet"
PRE_POST_BREAK_UTC = datetime(2025, 3, 4, 0, 0, tzinfo=UTC)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--zones", nargs="+", default=["DK1", "DE", "SE3", "NO2"])
    p.add_argument("--end", type=str, default="2025-03-03T23:00:00+00:00",
                   help="ISO UTC cutoff (default: end of DK1 train fold).")
    p.add_argument("--start", type=str, default="2016-01-01T00:00:00+00:00")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    df = load_trinity_prices(zones=tuple(args.zones))
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    df = df.filter((pl.col("utc_timestamp") >= start) & (pl.col("utc_timestamp") <= end))
    df = df.drop_nulls(["price_eur_mwh"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"wrote {args.out}: {df.height:,} rows across zones {args.zones}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
