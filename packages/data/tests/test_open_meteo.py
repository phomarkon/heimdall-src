from __future__ import annotations

from datetime import UTC, datetime

import pytest
from heimdall_data.open_meteo import WeatherLocation, fetch_weather_forecast, normalize_open_meteo


def test_normalize_open_meteo_maps_whitelisted_variables() -> None:
    payload = {
        "hourly": {
            "time": ["2026-05-11T00:00", "2026-05-11T01:00"],
            "temperature_2m": [10.0, 11.0],
            "wind_speed_10m": [5.0, 6.0],
            "shortwave_radiation": [100.0, 150.0],
        }
    }
    frame = normalize_open_meteo(payload, zone="DK1")
    assert list(frame.columns) == [
        "timestamp_utc",
        "zone",
        "temperature",
        "wind_speed",
        "solar_radiation",
    ]
    assert frame["zone"].tolist() == ["DK1", "DK1"]


def test_fetch_weather_rejects_unsupported_variable() -> None:
    with pytest.raises(ValueError, match="unsupported weather variables"):
        fetch_weather_forecast(
            WeatherLocation("DK1", 56.0, 9.0),
            variables=["moon_phase"],
            start=datetime(2026, 5, 11, tzinfo=UTC),
            end=datetime(2026, 5, 12, tzinfo=UTC),
        )


def test_fetch_weather_uses_session_and_window() -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "hourly": {
                    "time": ["2026-05-10T23:00", "2026-05-11T00:00", "2026-05-11T01:00"],
                    "temperature_2m": [9.0, 10.0, 11.0],
                }
            }

    class _Session:
        def __init__(self) -> None:
            self.params = {}

        def get(self, _url: str, *, params: dict, timeout: int) -> _Response:
            self.params = params
            assert timeout == 60
            return _Response()

    session = _Session()
    frame = fetch_weather_forecast(
        WeatherLocation("DK1", 56.0, 9.0),
        variables=["temperature"],
        start=datetime(2026, 5, 11, tzinfo=UTC),
        end=datetime(2026, 5, 11, 1, tzinfo=UTC),
        session=session,  # type: ignore[arg-type]
    )
    assert session.params["hourly"] == "temperature_2m"
    assert len(frame) == 2
