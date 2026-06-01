"""Tests for markets.load_markets. Network-free.

We craft a temporary markets root with a single fake parquet and verify:
- columns present (all 4) even when only one is on disk,
- forward-fill across an hourly grid,
- 18:00 UTC publication boundary respected (no future leak).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from heimdall_data.markets import MARKET_SERIES, load_markets


def _write_fake_series(root: Path, fname: str, dates: list[str], values: list[float]) -> None:
    pl.DataFrame(
        {
            "date": pl.Series([datetime.fromisoformat(d).date() for d in dates], dtype=pl.Date),
            "value": pl.Series(values, dtype=pl.Float64),
        }
    ).write_parquet(root / fname)


def test_all_columns_present_even_if_files_missing(tmp_path: Path) -> None:
    out = load_markets(
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 2, tzinfo=timezone.utc),
        root=tmp_path,
    )
    for col in MARKET_SERIES:
        assert col in out.columns
        assert out[col].is_null().all()


def test_forward_fill_from_publication_boundary(tmp_path: Path) -> None:
    _write_fake_series(
        tmp_path,
        "ttf_gas.parquet",
        ["2025-01-01", "2025-01-03"],
        [40.0, 45.0],
    )
    out = load_markets(
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 4, tzinfo=timezone.utc),
        root=tmp_path,
    )
    s = out.select("timestamp_utc", "gas_ttf_eur_mwh")
    # Before 2025-01-01 18:00 UTC there is no publication → null.
    assert s.filter(pl.col("timestamp_utc") == datetime(2025, 1, 1, 12, tzinfo=timezone.utc))[
        "gas_ttf_eur_mwh"
    ][0] is None
    # At/after 18:00 UTC on 2025-01-01 → 40.0.
    assert s.filter(pl.col("timestamp_utc") == datetime(2025, 1, 1, 18, tzinfo=timezone.utc))[
        "gas_ttf_eur_mwh"
    ][0] == 40.0
    # 2025-01-02 (no settle that day): still 40.0 (forward-filled).
    assert s.filter(pl.col("timestamp_utc") == datetime(2025, 1, 2, 12, tzinfo=timezone.utc))[
        "gas_ttf_eur_mwh"
    ][0] == 40.0
    # 2025-01-03 18:00 UTC → 45.0.
    assert s.filter(pl.col("timestamp_utc") == datetime(2025, 1, 3, 18, tzinfo=timezone.utc))[
        "gas_ttf_eur_mwh"
    ][0] == 45.0


def test_other_columns_untouched_when_only_one_file_present(tmp_path: Path) -> None:
    _write_fake_series(tmp_path, "ttf_gas.parquet", ["2025-01-01"], [40.0])
    out = load_markets(
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 2, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert not out["gas_ttf_eur_mwh"].is_null().all()
    for col in ("eu_ets_eur_t", "coal_api2_usd_t", "brent_usd_bbl"):
        assert out[col].is_null().all()
