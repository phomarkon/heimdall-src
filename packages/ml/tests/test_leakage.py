"""Pre/post-break leakage assertion tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from heimdall_ml.eval.leakage import (
    TEST_END_UTC,
    TEST_START_UTC,
    VAL_START_UTC,
    assert_no_test_overlap,
    assert_test_panel_only,
)


def _write_panel(path: Path, start: datetime, n: int) -> None:
    ts = [start + timedelta(minutes=15 * i) for i in range(n)]
    pl.DataFrame({"timestamp_utc": ts}).with_columns(
        pl.col("timestamp_utc").cast(pl.Datetime("us", time_zone="UTC"))
    ).write_parquet(path)


def test_train_assert_passes_for_pre_break(tmp_path: Path) -> None:
    p = tmp_path / "tr.parquet"
    _write_panel(p, datetime(2025, 1, 1, tzinfo=timezone.utc), 1000)
    assert_no_test_overlap(p, role="train")  # must not raise


def test_train_assert_fails_for_post_break(tmp_path: Path) -> None:
    p = tmp_path / "tr.parquet"
    # Train data ranging into val window — must fail
    _write_panel(p, datetime(2025, 3, 1, tzinfo=timezone.utc), 2000)
    with pytest.raises(ValueError, match="LEAKAGE"):
        assert_no_test_overlap(p, role="train")


def test_val_assert_fails_when_val_extends_into_test(tmp_path: Path) -> None:
    p = tmp_path / "va.parquet"
    # Val starts in the val window but extends past the test boundary.
    _write_panel(p, VAL_START_UTC, 50_000)
    with pytest.raises(ValueError, match="LEAKAGE"):
        assert_no_test_overlap(p, role="val")


def test_assert_test_panel_only_passes_for_test_window(tmp_path: Path) -> None:
    p = tmp_path / "te.parquet"
    _write_panel(p, TEST_START_UTC, 200)
    assert_test_panel_only(p)


def test_assert_test_panel_only_fails_for_val_window(tmp_path: Path) -> None:
    p = tmp_path / "te.parquet"
    _write_panel(p, VAL_START_UTC, 200)
    with pytest.raises(ValueError):
        assert_test_panel_only(p)
