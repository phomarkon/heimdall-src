"""Concatenate monthly DK1 panels into the train/val/test parquets.

Per docs/RESEARCH-PROPOSAL.md §5.7 the splits are deterministic at:

  - train  -> 2020-01-01 ... 2025-02-28        (≤ pre-break)
  - val    -> 2025-03-04 ... 2025-04-30
  - test   -> 2025-05-01 ... 2026-04-30        (sacred, evaluated once)

Pre-2025-03-04 the ENTSO-E A85 15-minute imbalance signal is unavailable, so
the synthesised ``imbalance_price_dkk_mwh_15min`` column falls back to the
hourly Energinet ``imbalance_price_dkk_mwh`` (already forward-filled onto the
15-min grid by ``loaders.load_dk1_panel``).

The script is idempotent: it overwrites the three named output parquets and
prints a summary. It does **not** make any network calls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"

TRAIN_END_UTC = datetime(2025, 3, 1, tzinfo=timezone.utc)
VAL_START_UTC = datetime(2025, 3, 4, tzinfo=timezone.utc)
VAL_END_UTC = datetime(2025, 5, 1, tzinfo=timezone.utc)
TEST_START_UTC = VAL_END_UTC
TEST_END_UTC = datetime(2026, 5, 1, tzinfo=timezone.utc)

# Columns the trainer expects (see apps/forecaster/train/dataset.py).
EXPECTED_COLS = (
    "timestamp_utc",
    "da_price_dkk_mwh",
    "load_actual_mw",
    "imbalance_price_dkk_mwh",
    "mfrr_up_volume_mw",
    "mfrr_down_volume_mw",
    "imbalance_price_dkk_mwh_15min",
)


def _gather_monthly_panels(in_dir: Path) -> pl.DataFrame:
    """Read every dk1_panel_<start>_<end>.parquet (monthly) and concat in order."""
    paths = sorted(in_dir.glob("dk1_panel_2*_2*.parquet"))
    # Skip the train/val/test artefacts themselves.
    paths = [p for p in paths if not p.stem.endswith(("_train", "_val", "_test"))]
    if not paths:
        raise FileNotFoundError(f"no monthly panels in {in_dir}")
    frames: list[pl.DataFrame] = []
    for p in paths:
        df = pl.read_parquet(p)
        # Earlier 2020-2024 monthlies have no A85 column; back-fill it from the
        # hourly Energinet column. Per docs/RESEARCH-PROPOSAL.md §4.4 the *target* is
        # the 15-min A85 price post-break; pre-break we use the hourly proxy as
        # the only available imbalance signal.
        if "imbalance_price_dkk_mwh_15min" not in df.columns:
            df = df.with_columns(
                pl.col("imbalance_price_dkk_mwh").alias("imbalance_price_dkk_mwh_15min"),
            )
        # Drop the A85-EUR helper column when present (we already carry the DKK form).
        keep = [c for c in EXPECTED_COLS if c in df.columns]
        df = df.select(keep)
        frames.append(df)
    out = pl.concat(frames, how="diagonal_relaxed").sort("timestamp_utc").unique(
        subset=["timestamp_utc"], keep="first", maintain_order=True
    )
    return out


def _split_and_write(panel: pl.DataFrame, out_dir: Path) -> dict[str, dict[str, str | int]]:
    train = panel.filter(pl.col("timestamp_utc") < TRAIN_END_UTC)
    val = panel.filter(
        (pl.col("timestamp_utc") >= VAL_START_UTC)
        & (pl.col("timestamp_utc") < VAL_END_UTC)
    )
    test = panel.filter(
        (pl.col("timestamp_utc") >= TEST_START_UTC)
        & (pl.col("timestamp_utc") < TEST_END_UTC)
    )
    summaries: dict[str, dict[str, str | int]] = {}
    for name, df in (("train", train), ("val", val), ("test", test)):
        # Drop rows with all-null target (training cannot use them).
        df = df.filter(pl.col("imbalance_price_dkk_mwh_15min").is_not_null())
        out = out_dir / f"dk1_panel_{name}.parquet"
        df.write_parquet(out)
        summaries[name] = {
            "rows": int(df.height),
            "start": str(df["timestamp_utc"].min()) if df.height else "",
            "end": str(df["timestamp_utc"].max()) if df.height else "",
            "path": str(out),
        }
    # Also write a single full panel for downstream KE2 / leakage checks.
    panel.write_parquet(out_dir / "dk1_panel.parquet")
    summaries["full"] = {
        "rows": int(panel.height),
        "start": str(panel["timestamp_utc"].min()),
        "end": str(panel["timestamp_utc"].max()),
        "path": str(out_dir / "dk1_panel.parquet"),
    }
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=PROCESSED)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED)
    args = parser.parse_args(argv)

    panel = _gather_monthly_panels(args.in_dir)
    summaries = _split_and_write(panel, args.out_dir)
    print("Built DK1 splits:")
    for k, v in summaries.items():
        print(f"  {k:5s}  rows={v['rows']:>7}  {v['start']}  ->  {v['end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
