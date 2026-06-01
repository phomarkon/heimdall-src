"""Market-context facade: tick orchestration for synthetic and real modes.

Data-access tools live in ``_market_data.py``; forecasting helpers live in
``_market_forecast.py``.  This module composes them into the two context
classes (``SyntheticMarketContext``, ``RealMarketContext``) and re-exports
everything for backward compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from heimdall_data.jao import empty_constraints_frame
from heimdall_data.entsoe import (
    get_cached_entsoe_flows,
    get_cached_entsoe_generation,
    get_cached_entsoe_loads,
    get_cached_entsoe_prices,
)
from heimdall_data.open_meteo import (
    WEATHER_VARIABLES,
    WeatherLocation,
    default_weather_window,
    get_cached_weather_forecast,
)
from heimdall_forecaster.inference import get_forecaster

from packages.simulator.forecast import ForecastMarketState, ForecastSource

# Re-export from sub-modules so existing ``from market_context import X`` works.
from heimdall_ai_society._market_data import (  # noqa: F401 — re-exports
    RealDataTools,
    _canonical_time,
    _iso_z,
)
from heimdall_ai_society._market_forecast import (
    fallback_forecast as _fallback_forecast,
    fit_baseline_forecaster as _fit_baseline_forecaster,
    forecast_for_tick as _forecast_for_tick,
    required_data_error as _required_data_error,
    _latest_numeric,
)


# ---------------------------------------------------------------------------
# Shared helper kept here because _load_* methods reference it and tests
# monkeypatch entsoe/weather names on *this* module.
# ---------------------------------------------------------------------------

def _concat_or_empty(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    available = [frame for frame in frames if not frame.empty]
    if not available:
        return pd.DataFrame(columns=columns)
    return _canonical_time(pd.concat(available, ignore_index=True, sort=False))


# ---------------------------------------------------------------------------
# TickContext
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TickContext:
    timestamp: datetime
    market_price_eur_mwh: float
    forecast: ForecastMarketState
    unavailable_reason: str | None = None


# ---------------------------------------------------------------------------
# SyntheticMarketContext
# ---------------------------------------------------------------------------

class SyntheticMarketContext:
    def __init__(self, *, seed: int, zone: str, start: datetime, forecaster_backend: str) -> None:
        self._rng = np.random.default_rng(seed)
        self._zone = zone
        self._now = start.astimezone(UTC)
        self._last_price = 65.0
        self._history = [60.0 + 4.0 * np.sin(i / 8.0) for i in range(192)]
        self._forecaster = get_forecaster(forecaster_backend, seed=seed)
        self._backend = forecaster_backend

    def next_tick(self) -> TickContext:
        shock = float(self._rng.normal(0.0, 4.0))
        self._last_price = 65.0 + 0.72 * (self._last_price - 65.0) + shock
        self._history.append(self._last_price)
        quantiles = self._forecaster.predict(self._history, horizon=1, levels=(0.1, 0.5, 0.9))[0]
        lower, median, upper = (float(v) for v in quantiles.values)
        lower, median, upper = sorted((lower, median, upper))
        forecast = ForecastMarketState(
            delivery_timestamp=_iso_z(self._now),
            zone=self._zone,
            issued_at=_iso_z(self._now - timedelta(minutes=50)),
            activation_direction="up",
            activation_volume_mwh=8.0,
            spot_price_eur_mwh=round(self._last_price - 3.0, 6),
            imbalance_price_lower_eur_mwh=round(lower, 6),
            imbalance_price_median_eur_mwh=round(median, 6),
            imbalance_price_upper_eur_mwh=round(upper, 6),
            mfrr_up_price_lower_eur_mwh=round(lower, 6),
            mfrr_up_price_median_eur_mwh=round(median, 6),
            mfrr_up_price_upper_eur_mwh=round(upper, 6),
            mfrr_down_price_lower_eur_mwh=round(lower - 12.0, 6),
            mfrr_down_price_median_eur_mwh=round(median - 12.0, 6),
            mfrr_down_price_upper_eur_mwh=round(upper - 12.0, 6),
            source=ForecastSource(
                kind="baseline_conformal",
                window_start=_iso_z(self._now - timedelta(days=2)),
                method=self._backend,
            ),
        )
        ctx = TickContext(
            timestamp=self._now,
            market_price_eur_mwh=round(self._last_price, 6),
            forecast=forecast,
        )
        self._now = self._now + timedelta(minutes=15)
        return ctx

    def tools_for_tick(self, tick: TickContext) -> RealDataTools | None:
        return None

    def tick_for_forecaster(self, tick: TickContext, backend: str) -> TickContext:
        return tick


# ---------------------------------------------------------------------------
# RealMarketContext
# ---------------------------------------------------------------------------

class RealMarketContext:
    def __init__(
        self,
        *,
        zone: str,
        start: datetime,
        data_start: datetime | None,
        data_end: datetime | None,
        default_lookback_hours: int,
        cache_refresh: bool,
        weather_locations: dict[str, Any],
        forecaster_backend: str = "f0",
        seed: int = 42,
        context_dataset_dir: Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._zone = zone
        self._now = start.astimezone(UTC)
        self._default_lookback_hours = default_lookback_hours
        self._cache_refresh = cache_refresh
        self._data_start = (data_start or (self._now - timedelta(hours=max(48, default_lookback_hours)))).astimezone(UTC)
        self._data_end = (data_end or (self._now + timedelta(hours=48))).astimezone(UTC)
        self._cache_dir = cache_dir
        self._context_dataset_dir = context_dataset_dir
        self._forecaster_backend = forecaster_backend
        self._seed = seed
        self._weather_locations = {
            key: WeatherLocation(zone=key, latitude=float(value.latitude), longitude=float(value.longitude))
            for key, value in weather_locations.items()
        }
        prepared = self._load_prepared_dataset()
        if prepared is not None:
            self._prices, self._loads, self._generation, self._flows, self._weather, self._outages, self._jao_constraints = prepared
        else:
            self._prices = self._load_prices()
            self._loads = self._load_loads()
            self._generation = self._load_generation()
            self._flows = self._load_flows()
            self._weather = self._load_weather()
            self._outages: list[dict[str, Any]] = []
            self._jao_constraints = empty_constraints_frame()
        self._baseline_forecaster = _fit_baseline_forecaster(self._prices)
        self._zoo_forecasters: dict[str, Any] = {}
        if forecaster_backend != "f0":
            self._zoo_forecasters[forecaster_backend] = get_forecaster(forecaster_backend, seed=seed)

    # -- Tick interface -----------------------------------------------------

    def next_tick(self) -> TickContext:
        unavailable = _required_data_error(self._prices, self._zone, self._now)
        if unavailable is not None:
            forecast = _fallback_forecast(self._now, self._zone, unavailable)
            ctx = TickContext(self._now, forecast.spot_price_eur_mwh, forecast, unavailable_reason=unavailable)
            self._now += timedelta(minutes=15)
            return ctx
        forecast = _forecast_for_tick(
            timestamp=self._now,
            backend=self._forecaster_backend,
            zone=self._zone,
            prices=self._prices,
            loads=self._loads,
            baseline_forecaster=self._baseline_forecaster,
            zoo_forecasters=self._zoo_forecasters,
            seed=self._seed,
        )
        price = _latest_numeric(
            self._prices[
                (self._prices["zone"] == self._zone)
                & (self._prices["price_type"].isin(["imbalance", "day_ahead"]))
                & (self._prices["timestamp_utc"] <= pd.Timestamp(self._now))
            ],
            "price_eur_mwh",
            default=forecast.spot_price_eur_mwh,
        )
        ctx = TickContext(self._now, price, forecast)
        self._now += timedelta(minutes=15)
        return ctx

    def tick_for_forecaster(self, tick: TickContext, backend: str) -> TickContext:
        forecast = _forecast_for_tick(
            timestamp=tick.timestamp,
            backend=backend,
            zone=self._zone,
            prices=self._prices,
            loads=self._loads,
            baseline_forecaster=self._baseline_forecaster,
            zoo_forecasters=self._zoo_forecasters,
            seed=self._seed,
        )
        return TickContext(tick.timestamp, tick.market_price_eur_mwh, forecast, tick.unavailable_reason)

    def tools_for_tick(self, tick: TickContext) -> RealDataTools:
        return RealDataTools(
            now=tick.timestamp,
            zone=self._zone,
            prices=self._prices,
            loads=self._loads,
            generation=self._generation,
            flows=self._flows,
            weather=self._weather,
            outages=self._outages,
            jao_constraints=self._jao_constraints,
            default_lookback_hours=self._default_lookback_hours,
        )

    # -- Data loading (kept here so tests can monkeypatch entsoe/weather
    #    names on *this* module) -----------------------------------------

    def _load_prepared_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame] | None:
        if self._context_dataset_dir is None:
            return None
        root = self._context_dataset_dir
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"prepared context dataset missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("visibility") != "agent_context":
            raise RuntimeError(
                f"prepared dataset {root} is not agent_context visibility: {manifest.get('visibility')!r}"
            )
        required = ["prices.parquet", "loads.parquet", "generation.parquet", "flows.parquet", "weather.parquet", "outages.json"]
        missing = [name for name in required if not (root / name).exists()]
        if missing:
            raise RuntimeError(f"prepared real dataset missing files in {root}: {missing}")
        outages = json.loads((root / "outages.json").read_text(encoding="utf-8"))
        if not isinstance(outages, list):
            raise RuntimeError(f"prepared outage file is not a list: {root / 'outages.json'}")
        jao_path = root / "jao_constraints.parquet"
        jao_constraints = pd.read_parquet(jao_path) if jao_path.exists() else empty_constraints_frame()
        return (
            pd.read_parquet(root / "prices.parquet"),
            pd.read_parquet(root / "loads.parquet"),
            pd.read_parquet(root / "generation.parquet"),
            pd.read_parquet(root / "flows.parquet"),
            pd.read_parquet(root / "weather.parquet"),
            outages,
            jao_constraints,
        )

    def _load_prices(self) -> pd.DataFrame:
        frames = []
        for price_type in ["day_ahead", "imbalance", "mfrr_up", "mfrr_down"]:
            try:
                frames.append(get_cached_entsoe_prices(zone=self._zone, price_type=price_type, start=self._data_start, end=self._data_end, refresh=self._cache_refresh, cache_dir=self._cache_dir).frame)
            except Exception:
                continue
        return _concat_or_empty(frames, ["timestamp_utc", "zone", "price_type", "price_eur_mwh"])

    def _load_loads(self) -> pd.DataFrame:
        frames = []
        for kind in ["actual", "forecast"]:
            try:
                frames.append(get_cached_entsoe_loads(zone=self._zone, kind=kind, start=self._data_start, end=self._data_end, refresh=self._cache_refresh, cache_dir=self._cache_dir).frame)
            except Exception:
                continue
        return _concat_or_empty(frames, ["timestamp_utc", "zone", "kind", "load_mw"])

    def _load_generation(self) -> pd.DataFrame:
        frames = []
        for generation_type in ["wind", "solar", "hydro", "thermal"]:
            try:
                frames.append(get_cached_entsoe_generation(zone=self._zone, generation_type=generation_type, start=self._data_start, end=self._data_end, refresh=self._cache_refresh, cache_dir=self._cache_dir).frame)
            except Exception:
                continue
        return _concat_or_empty(frames, ["timestamp_utc", "zone", "generation_type", "production_type", "generation_mw"])

    def _load_flows(self) -> pd.DataFrame:
        counterparties = ["DK_2"] if self._zone == "DK1" else ["DK_1"]
        frames = []
        for counterparty in counterparties:
            try:
                frames.append(get_cached_entsoe_flows(zone=self._zone, counterparty=counterparty, start=self._data_start, end=self._data_end, refresh=self._cache_refresh, cache_dir=self._cache_dir).frame)
            except Exception:
                continue
        return _concat_or_empty(frames, ["timestamp_utc", "from_zone", "to_zone", "flow_mw"])

    def _load_weather(self) -> pd.DataFrame:
        location = self._weather_locations.get(self._zone)
        if location is None:
            return _concat_or_empty([], ["timestamp_utc", "zone", *WEATHER_VARIABLES])
        start, end = default_weather_window(self._now, horizon_hours=72)
        try:
            return get_cached_weather_forecast(location, variables=list(WEATHER_VARIABLES), start=start, end=end, refresh=self._cache_refresh, cache_dir=self._cache_dir).frame
        except Exception:
            return _concat_or_empty([], ["timestamp_utc", "zone", *WEATHER_VARIABLES])
