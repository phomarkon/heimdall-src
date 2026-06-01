from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from heimdall_data.cache import CachedFrame, read_cached_frame, write_cached_frame

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

WEATHER_VARIABLES: dict[str, str] = {
    "temperature": "temperature_2m",
    "wind_speed": "wind_speed_10m",
    "wind_direction": "wind_direction_10m",
    "wind_gusts": "wind_gusts_10m",
    "solar_radiation": "shortwave_radiation",
    "cloud_cover": "cloud_cover",
    "precipitation": "precipitation",
    "pressure": "surface_pressure",
    "humidity": "relative_humidity_2m",
}


@dataclass(frozen=True)
class WeatherLocation:
    zone: str
    latitude: float
    longitude: float


class WeatherDataError(RuntimeError):
    pass


def validate_weather_variables(variables: list[str] | tuple[str, ...]) -> list[str]:
    unsupported = sorted(set(variables) - set(WEATHER_VARIABLES))
    if unsupported:
        raise ValueError(f"unsupported weather variables: {unsupported}")
    return list(dict.fromkeys(variables))


def fetch_weather_forecast(
    location: WeatherLocation,
    *,
    variables: list[str] | tuple[str, ...],
    start: datetime,
    end: datetime,
    timeout_seconds: int = 60,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    selected = validate_weather_variables(variables)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("weather windows must be timezone-aware")
    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "hourly": ",".join(WEATHER_VARIABLES[v] for v in selected),
        "start_date": start_utc.date().isoformat(),
        "end_date": end_utc.date().isoformat(),
        "timezone": "UTC",
    }
    payload = _get_json(session or requests.Session(), params=params, timeout=timeout_seconds)
    frame = normalize_open_meteo(payload, zone=location.zone)
    mask = (frame["timestamp_utc"] >= pd.Timestamp(start_utc)) & (
        frame["timestamp_utc"] <= pd.Timestamp(end_utc)
    )
    columns = ["timestamp_utc", "zone", *selected]
    return frame.loc[mask, columns].reset_index(drop=True)


def get_cached_weather_forecast(
    location: WeatherLocation,
    *,
    variables: list[str] | tuple[str, ...],
    start: datetime,
    end: datetime,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> CachedFrame:
    selected = validate_weather_variables(variables)
    key = (
        f"open_meteo_{location.zone}_{start.astimezone(UTC).date().isoformat()}_"
        f"{end.astimezone(UTC).date().isoformat()}_{'-'.join(selected)}"
    )
    if not refresh:
        cached = read_cached_frame(key, cache_dir=cache_dir)
        if cached is not None:
            return cached
    frame = fetch_weather_forecast(location, variables=selected, start=start, end=end)
    if frame.empty:
        raise WeatherDataError(f"Open-Meteo returned no rows for {location.zone}")
    return write_cached_frame(
        key,
        frame,
        source="open_meteo",
        cache_dir=cache_dir,
        metadata={
            "zone": location.zone,
            "variables": selected,
            "window_start_utc": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "window_end_utc": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        },
    )


def normalize_open_meteo(payload: dict[str, Any], *, zone: str) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise WeatherDataError("Open-Meteo payload missing hourly.time")
    timestamps = pd.to_datetime(hourly["time"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise WeatherDataError("Open-Meteo payload contains invalid timestamps")
    data: dict[str, Any] = {"timestamp_utc": timestamps, "zone": zone}
    reverse = {v: k for k, v in WEATHER_VARIABLES.items()}
    for api_name, values in hourly.items():
        if api_name == "time":
            continue
        canonical = reverse.get(api_name)
        if canonical is not None:
            data[canonical] = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.DataFrame(data).sort_values("timestamp_utc").reset_index(drop=True)


def fetch_previous_runs_forecast(
    location: WeatherLocation,
    *,
    variables: list[str] | tuple[str, ...],
    start: datetime,
    end: datetime,
    lead_days: tuple[int, ...] = (1, 2, 3),
    timeout_seconds: int = 60,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    selected = validate_weather_variables(variables)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("weather windows must be timezone-aware")
    if any(d < 1 or d > 7 for d in lead_days):
        raise ValueError("lead_days must be in [1, 7]")
    start_utc = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    hourly = []
    for var in selected:
        api_name = WEATHER_VARIABLES[var]
        for d in lead_days:
            hourly.append(f"{api_name}_previous_day{d}")
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "hourly": ",".join(hourly),
        "start_date": start_utc.date().isoformat(),
        "end_date": end_utc.date().isoformat(),
        "timezone": "UTC",
    }
    payload = _get_json(
        session or requests.Session(),
        params=params,
        timeout=timeout_seconds,
        base_url=OPEN_METEO_PREVIOUS_RUNS_URL,
    )
    frame = normalize_open_meteo_previous_runs(payload, zone=location.zone, lead_days=lead_days)
    mask = (frame["timestamp_utc"] >= pd.Timestamp(start_utc)) & (
        frame["timestamp_utc"] <= pd.Timestamp(end_utc)
    )
    return frame.loc[mask].reset_index(drop=True)


def get_cached_previous_runs_forecast(
    location: WeatherLocation,
    *,
    variables: list[str] | tuple[str, ...],
    start: datetime,
    end: datetime,
    lead_days: tuple[int, ...] = (1, 2, 3),
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> CachedFrame:
    selected = validate_weather_variables(variables)
    lead_key = "-".join(str(d) for d in lead_days)
    key = (
        f"open_meteo_prev_runs_{location.zone}_{start.astimezone(UTC).date().isoformat()}_"
        f"{end.astimezone(UTC).date().isoformat()}_{'-'.join(selected)}_d{lead_key}"
    )
    if not refresh:
        cached = read_cached_frame(key, cache_dir=cache_dir)
        if cached is not None:
            return cached
    frame = fetch_previous_runs_forecast(
        location,
        variables=selected,
        start=start,
        end=end,
        lead_days=lead_days,
    )
    if frame.empty:
        raise WeatherDataError(f"Open-Meteo previous runs returned no rows for {location.zone}")
    return write_cached_frame(
        key,
        frame,
        source="open_meteo_previous_runs",
        cache_dir=cache_dir,
        metadata={
            "zone": location.zone,
            "variables": selected,
            "lead_days": list(lead_days),
            "window_start_utc": start.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "window_end_utc": end.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        },
    )


def normalize_open_meteo_previous_runs(
    payload: dict[str, Any],
    *,
    zone: str,
    lead_days: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise WeatherDataError("Open-Meteo previous-runs payload missing hourly.time")
    timestamps = pd.to_datetime(hourly["time"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise WeatherDataError("Open-Meteo previous-runs payload contains invalid timestamps")
    out: dict[str, Any] = {"timestamp_utc": timestamps, "zone": zone}
    for canonical, api_name in WEATHER_VARIABLES.items():
        _ = canonical
        for d in lead_days:
            key = f"{api_name}_previous_day{d}"
            values = hourly.get(key)
            if values is None:
                continue
            out[f"wx_prev{d}_{api_name}"] = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.DataFrame(out).sort_values("timestamp_utc").reset_index(drop=True)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _get_json(
    session: requests.Session, *, params: dict[str, Any], timeout: int, base_url: str = OPEN_METEO_URL
) -> dict[str, Any]:
    response = session.get(base_url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise WeatherDataError("Open-Meteo response was not a JSON object")
    return payload


def default_weather_window(now: datetime, horizon_hours: int = 48) -> tuple[datetime, datetime]:
    current = now.astimezone(UTC)
    start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, current + timedelta(hours=horizon_hours)


__all__ = [
    "OPEN_METEO_URL",
    "OPEN_METEO_PREVIOUS_RUNS_URL",
    "WEATHER_VARIABLES",
    "WeatherDataError",
    "WeatherLocation",
    "default_weather_window",
    "fetch_previous_runs_forecast",
    "fetch_weather_forecast",
    "get_cached_previous_runs_forecast",
    "get_cached_weather_forecast",
    "normalize_open_meteo_previous_runs",
    "normalize_open_meteo",
    "validate_weather_variables",
]
