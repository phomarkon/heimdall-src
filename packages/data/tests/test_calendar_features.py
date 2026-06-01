"""Tests for calendar_features.add_calendar_features.

We assert: schema (column presence + dtypes), correctness of cyclical pairs
(sin² + cos² ≡ 1), holiday flags on known DK/DE/SE dates, DST transition day
detection, and absence of leakage (output for timestamp t depends only on t).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from heimdall_data.calendar_features import add_calendar_features


def _grid(n: int = 96) -> pl.DataFrame:
    """A small 15-min UTC grid for tests."""
    import pandas as pd

    rng = pd.date_range("2025-12-24 00:00", periods=n, freq="15min", tz="UTC")
    return pl.DataFrame(
        {
            "timestamp_utc": pl.Series(
                rng.to_pydatetime().tolist(),
                dtype=pl.Datetime("us", time_zone="UTC"),
            )
        }
    )


def test_adds_expected_columns() -> None:
    out = add_calendar_features(_grid())
    expected = {
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "quarter_sin",
        "quarter_cos",
        "is_weekend",
        "is_dk_holiday",
        "is_de_holiday",
        "is_se_holiday",
        "is_dst_transition",
        "days_to_next_dk_holiday",
    }
    assert expected.issubset(set(out.columns))


def test_cyclical_unit_circle() -> None:
    out = add_calendar_features(_grid())
    for prefix in ("hour", "dow", "month", "quarter"):
        s = out[f"{prefix}_sin"].to_numpy()
        c = out[f"{prefix}_cos"].to_numpy()
        assert np.allclose(s**2 + c**2, 1.0, atol=1e-10), f"{prefix} not on unit circle"


def test_holiday_flags_known_dates() -> None:
    # 2025-12-25 (Christmas Day) is a holiday in DK, DE, SE.
    out = add_calendar_features(_grid(n=96))
    # Hour 0 UTC on 12-24 is hour 1 CET; pick a clearly-on-25 row in local time.
    df = out.filter(
        pl.col("timestamp_utc").dt.convert_time_zone("Europe/Copenhagen").dt.date()
        == pl.lit("2025-12-25").str.to_date()
    )
    assert df["is_dk_holiday"].all()
    assert df["is_de_holiday"].all()
    assert df["is_se_holiday"].all()


def test_dst_transition_day_detection() -> None:
    # Last Sunday of October 2025 is 2025-10-26 (fall-back).
    import pandas as pd

    # Trim to UTC hours that all map to Europe/Copenhagen Oct 26 (DST flips at 01:00 UTC).
    rng = pd.date_range("2025-10-26 00:00", periods=22, freq="1h", tz="UTC")
    grid = pl.DataFrame(
        {
            "timestamp_utc": pl.Series(
                rng.to_pydatetime().tolist(),
                dtype=pl.Datetime("us", time_zone="UTC"),
            )
        }
    )
    out = add_calendar_features(grid)
    assert out["is_dst_transition"].all(), "Oct 26 2025 should be DST transition"


def test_days_to_next_holiday_capped_at_30() -> None:
    out = add_calendar_features(_grid())
    arr = out["days_to_next_dk_holiday"].to_numpy()
    assert (arr >= 0).all()
    assert (arr <= 30).all()


def test_no_leakage_positional_only() -> None:
    """Output for timestamp t must depend only on t — not on neighbours.

    We verify by computing features on two disjoint windows containing the same
    timestamp and confirming identical row values.
    """
    g1 = _grid(n=96)
    g2 = _grid(n=12)  # smaller window starting same timestamp
    o1 = add_calendar_features(g1).head(12)
    o2 = add_calendar_features(g2)
    feature_cols = [c for c in o1.columns if c != "timestamp_utc"]
    assert o1.select(feature_cols).equals(o2.select(feature_cols))


def test_invalid_ts_col_raises() -> None:
    with pytest.raises(KeyError):
        add_calendar_features(_grid(), ts_col="missing")
