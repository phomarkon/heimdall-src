"""Loader tests — both clients are stubbed; no network access."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from heimdall_data import loaders


class _StubEntsoe:
    def __init__(self) -> None:
        idx = pd.date_range("2025-03-04", periods=4, freq="15min", tz="UTC")
        self._da = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)
        self._load = pd.Series([1500.0, 1510.0, 1520.0, 1530.0], index=idx)
        self._imb = pd.DataFrame({"Long": [40.0, 41.0, 42.0, 43.0], "Short": [40.0, 41.0, 42.0, 43.0]}, index=idx)

    def day_ahead_prices(self, start, end):  # type: ignore[no-untyped-def]
        return self._da

    def total_load(self, start, end, kind="actual"):  # type: ignore[no-untyped-def]
        return self._load

    def imbalance_prices(self, start, end):  # type: ignore[no-untyped-def]
        return self._imb


class _StubEnerginet:
    def regulating_balance(self, start, end, area="DK1"):  # type: ignore[no-untyped-def]
        return [
            {
                "HourUTC": "2025-03-04T00:00:00",
                "ImbalancePriceDKK": 200.0,
                "mFRRUpActBal": 5.0,
                "mFRRDownActBal": 0.0,
            },
            {
                "HourUTC": "2025-03-04T00:15:00",
                "ImbalancePriceDKK": 210.0,
                "mFRRUpActBal": 3.0,
                "mFRRDownActBal": 0.0,
            },
            {
                "HourUTC": "2025-03-04T00:30:00",
                "ImbalancePriceDKK": 220.0,
                "mFRRUpActBal": 0.0,
                "mFRRDownActBal": 1.0,
            },
            {
                "HourUTC": "2025-03-04T00:45:00",
                "ImbalancePriceDKK": 230.0,
                "mFRRUpActBal": 0.0,
                "mFRRDownActBal": 2.0,
            },
        ]


def test_load_dk1_panel_aligns_to_quarter_grid(tmp_path: Path) -> None:
    start = datetime(2025, 3, 4, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 3, 4, 1, 0, tzinfo=timezone.utc)
    panel = loaders.load_dk1_panel(
        start,
        end,
        split="post",
        entsoe=_StubEntsoe(),  # type: ignore[arg-type]
        energinet=_StubEnerginet(),  # type: ignore[arg-type]
        cache_dir=tmp_path,
    )
    # 4 quarter-hours over the 1-hour window
    assert panel.height == 4
    cols = set(panel.columns)
    assert {
        "timestamp_utc",
        "imbalance_price_dkk_mwh",
        "mfrr_up_volume_mw",
        "mfrr_down_volume_mw",
        "da_price_dkk_mwh",
        "load_actual_mw",
    } <= cols
    # Persisted to the cache_dir
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1


def test_split_pre_disallows_post_break_window() -> None:
    with pytest.raises(ValueError, match="end is after"):
        loaders.load_dk1_panel(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 4, 1, tzinfo=timezone.utc),
            split="pre",
            entsoe=_StubEntsoe(),  # type: ignore[arg-type]
            energinet=_StubEnerginet(),  # type: ignore[arg-type]
        )


def test_split_post_disallows_pre_break_window() -> None:
    with pytest.raises(ValueError, match="start is before"):
        loaders.load_dk1_panel(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 4, 1, tzinfo=timezone.utc),
            split="post",
            entsoe=_StubEntsoe(),  # type: ignore[arg-type]
            energinet=_StubEnerginet(),  # type: ignore[arg-type]
        )


def test_naive_timestamps_rejected() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        loaders.load_dk1_panel(
            datetime(2025, 3, 4),
            datetime(2025, 3, 5, tzinfo=timezone.utc),
            entsoe=_StubEntsoe(),  # type: ignore[arg-type]
            energinet=_StubEnerginet(),  # type: ignore[arg-type]
        )


def test_panel_handles_empty_energinet_response(tmp_path: Path) -> None:
    class _Empty(_StubEnerginet):
        def regulating_balance(self, *a, **kw):  # type: ignore[no-untyped-def]
            return []

    panel = loaders.load_dk1_panel(
        datetime(2025, 3, 4, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 3, 4, 1, 0, tzinfo=timezone.utc),
        split="post",
        entsoe=_StubEntsoe(),  # type: ignore[arg-type]
        energinet=_Empty(),  # type: ignore[arg-type]
        cache_dir=tmp_path,
    )
    # The Energinet columns should be present but null
    assert panel["imbalance_price_dkk_mwh"].null_count() == 4
