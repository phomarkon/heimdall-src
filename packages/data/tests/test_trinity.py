"""Tests for the Trinity vendored-features loader.

These run against the actual data/external/trinity/*.parquet files committed
via the one-shot import on 2026-05-16 (commit 7dccf21e).

Tests are SKIPPED when the vendored data is absent (e.g. fresh clone before
running tools/get_data.sh to re-import).
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from heimdall_data.trinity import (
    DEFAULT_WEATHER_LAG_HOURS,
    TRINITY_ROOT,
    load_trinity_prices,
    load_trinity_weather,
    make_dk1_wide_features,
)

pytestmark = pytest.mark.skipif(
    not TRINITY_ROOT.exists() or not (TRINITY_ROOT / "weather.parquet").exists(),
    reason="Trinity vendored parquets not present; see data/external/trinity/PROVENANCE.md",
)


def test_prices_dk1_loads_and_is_sorted() -> None:
    df = load_trinity_prices(zones=("DK1",))
    assert "price_eur_mwh" in df.columns
    assert df.height > 0
    ts = df["utc_timestamp"]
    assert ts.is_sorted()


def test_weather_lag_applied_by_default() -> None:
    """With default lag (24h), the earliest timestamp must be 24h later than raw."""
    lagged = load_trinity_weather(zones=("DK1",), lag_hours=DEFAULT_WEATHER_LAG_HOURS)
    raw = load_trinity_weather(zones=("DK1",), lag_hours=0)
    assert lagged.height == raw.height
    delta = lagged["utc_timestamp"][0] - raw["utc_timestamp"][0]
    assert delta.total_seconds() == DEFAULT_WEATHER_LAG_HOURS * 3600


def test_weather_lag_zero_escape_hatch() -> None:
    """lag_hours=0 must work (M3 leakage-quantification retrain only)."""
    raw = load_trinity_weather(zones=("DK1",), lag_hours=0)
    assert raw.height > 0


def test_weather_negative_lag_rejected() -> None:
    with pytest.raises(ValueError):
        load_trinity_weather(zones=("DK1",), lag_hours=-1)


def test_make_dk1_wide_features_schema() -> None:
    out = make_dk1_wide_features(
        start_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 1, 7, tzinfo=timezone.utc),
    )
    expected = {
        "timestamp_utc",
        "dk1_da_price_eur_mwh",
        "de_da_price_eur_mwh",
        "se3_da_price_eur_mwh",
        "no2_da_price_eur_mwh",
        "dk1_load_mw",
        "wind_gen_mw",
        "solar_gen_mw",
        "fossil_gen_mw",
        "flow_dk1_de_mw",
        "flow_dk1_se3_mw",
        "flow_dk1_no2_mw",
        "wind_speed_100m",
        "temperature_2m",
        "shortwave_radiation",
    }
    missing = expected - set(out.columns)
    assert not missing, f"missing columns: {missing}"
    assert out.height > 0
    # Non-trivial weather: should have at least some non-null wind speed.
    assert out["wind_speed_100m"].is_not_null().sum() > 0


def test_window_filter_respected() -> None:
    s = datetime(2024, 6, 1, tzinfo=timezone.utc)
    e = datetime(2024, 6, 2, tzinfo=timezone.utc)
    out = make_dk1_wide_features(start_utc=s, end_utc=e)
    assert out["timestamp_utc"].min() >= s
    assert out["timestamp_utc"].max() < e
