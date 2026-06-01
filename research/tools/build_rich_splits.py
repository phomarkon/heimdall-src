"""Split the rich panel into train/val/test parquets aligned to the existing
canonical splits (2025-02-28 / 2025-03-04 / 2025-05-01).

Output:
  - data/processed/dk1_panel_rich_train.parquet  (<= 2025-02-28 23:45 UTC)
  - data/processed/dk1_panel_rich_val.parquet    (2025-03-04 → 2025-04-30 23:45)
  - data/processed/dk1_panel_rich_test.parquet   (2025-05-01 → 2026-04-29 23:45)

Anomaly-features parquet is split with the same boundaries so train/val/test
joins are pairwise consistent.

Also emits the `_archive` variants for the M3 leakage-quantification retrain.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from heimdall_data.rich_panel import (
    build_rich_panel,
)
from heimdall_ml.features.anomaly import build_anomaly_features

PRE_POST = datetime(2025, 3, 4, 0, 0, tzinfo=UTC)
TEST_START = datetime(2025, 5, 1, 0, 0, tzinfo=UTC)
TEST_END = datetime(2026, 4, 30, 0, 0, tzinfo=UTC)

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def _slice_and_write(panel: pl.DataFrame, stem: str, essential_cols: list[str] | None = None) -> None:
    """Slice the panel into train/val/test and fill non-essential nulls with 0.

    ``essential_cols`` are columns that, if null, must remove the row. Defaults
    to the target column for rich panels. Other nulls (UMM pre-2024, markets if
    fetch script not run, anomaly tail-effects) are zero-filled — sliding-window
    models tolerate this and downstream feature-importance will recognise zero
    columns as low-signal.
    """
    if essential_cols is None:
        # Use ONLY the 15-min imbalance target — the hourly fallback column is
        # null in the test window (test set is post-2025-05; we use A85 15-min
        # natively).
        essential_cols = [c for c in ("imbalance_price_dkk_mwh_15min",) if c in panel.columns]

    def _slice(df: pl.DataFrame, start, end) -> pl.DataFrame:
        s = df.filter(
            (pl.col("timestamp_utc") >= start) & (pl.col("timestamp_utc") < end)
        )
        if essential_cols:
            s = s.drop_nulls(essential_cols)
        # Fill all remaining nulls with 0.0 (numeric) / False (bool) — downstream OK.
        return s.with_columns(
            [
                pl.col(c).fill_null(0.0) if dtype.is_numeric() else pl.col(c).fill_null(False)
                for c, dtype in zip(s.columns, s.dtypes, strict=True)
                if c != "timestamp_utc"
            ]
        )

    train = _slice(panel, panel["timestamp_utc"].min(), PRE_POST)
    val = _slice(panel, PRE_POST, TEST_START)
    test = _slice(panel, TEST_START, TEST_END)
    train.write_parquet(PROCESSED / f"{stem}_train.parquet")
    val.write_parquet(PROCESSED / f"{stem}_val.parquet")
    test.write_parquet(PROCESSED / f"{stem}_test.parquet")
    print(f"{stem}: train={train.height:,} val={val.height:,} test={test.height:,}")


def main() -> None:
    rich = build_rich_panel()
    _slice_and_write(rich, "dk1_panel_rich")

    # M3 archive variant: unlagged weather, for F8b-archive leakage quantification.
    rich_arch = build_rich_panel(weather_lag_hours=0, rebuild=True)
    _slice_and_write(rich_arch, "dk1_panel_rich_archive")

    anom = build_anomaly_features(rich_panel=rich)
    _slice_and_write(anom, "anomaly_features")


if __name__ == "__main__":
    main()
