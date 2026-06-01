"""Forecasting helpers for ``RealMarketContext``.

Extracted from ``market_context.py`` to keep data-loading and tick
orchestration separate from the baseline-forecaster fitting and
zoo-forecaster prediction logic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from heimdall_forecaster.inference import get_forecaster

from packages.simulator.forecast import BaselineMFRRForecaster, ForecastMarketState, ForecastSource

from heimdall_ai_society._market_data import _iso_z


def fit_baseline_forecaster(prices: pd.DataFrame) -> BaselineMFRRForecaster:
    history = _forecast_history(prices)
    if history.empty:
        raise RuntimeError("real market context requires cached day-ahead or imbalance prices")
    return BaselineMFRRForecaster.fit(history, lookback_days=7)


def forecast_for_tick(
    *,
    timestamp: datetime,
    backend: str,
    zone: str,
    prices: pd.DataFrame,
    loads: pd.DataFrame,
    baseline_forecaster: BaselineMFRRForecaster,
    zoo_forecasters: dict[str, Any],
    seed: int,
) -> ForecastMarketState:
    if backend == "f0":
        return baseline_forecaster.forecast(
            delivery_timestamp=_iso_z(timestamp),
            zone=zone,
            issued_at=_iso_z(timestamp - timedelta(minutes=50)),
        )
    issued_at = timestamp - timedelta(minutes=50)
    history = _zoo_history(prices, loads, issued_at, zone, backend)
    forecaster = zoo_forecasters.get(backend)
    if forecaster is None:
        forecaster = get_forecaster(backend, seed=seed)
        zoo_forecasters[backend] = forecaster
    quantiles = forecaster.predict(history, horizon=1, levels=(0.1, 0.5, 0.9))[0]
    lower, median, upper = sorted(float(v) for v in quantiles.values)
    spot = _latest_numeric(
        prices[
            (prices["zone"] == zone)
            & (prices["price_type"] == "day_ahead")
            & (prices["timestamp_utc"] <= pd.Timestamp(issued_at))
        ],
        "price_eur_mwh",
        default=median,
    )
    return ForecastMarketState(
        delivery_timestamp=_iso_z(timestamp),
        zone=zone,
        issued_at=_iso_z(issued_at),
        activation_direction="up",
        activation_volume_mwh=0.0,
        spot_price_eur_mwh=round(spot, 6),
        imbalance_price_lower_eur_mwh=round(lower, 6),
        imbalance_price_median_eur_mwh=round(median, 6),
        imbalance_price_upper_eur_mwh=round(upper, 6),
        mfrr_up_price_lower_eur_mwh=round(lower, 6),
        mfrr_up_price_median_eur_mwh=round(median, 6),
        mfrr_up_price_upper_eur_mwh=round(upper, 6),
        mfrr_down_price_lower_eur_mwh=round(lower, 6),
        mfrr_down_price_median_eur_mwh=round(median, 6),
        mfrr_down_price_upper_eur_mwh=round(upper, 6),
        source=ForecastSource(
            kind="zoo_quantile",
            window_start=_iso_z(issued_at - timedelta(hours=48)),
            method=backend,
        ),
    )


def fallback_forecast(timestamp: datetime, zone: str, reason: str) -> ForecastMarketState:
    return ForecastMarketState(
        delivery_timestamp=_iso_z(timestamp),
        zone=zone,
        issued_at=_iso_z(timestamp - timedelta(minutes=50)),
        activation_direction="neutral",
        activation_volume_mwh=0.0,
        spot_price_eur_mwh=0.0,
        imbalance_price_lower_eur_mwh=0.0,
        imbalance_price_median_eur_mwh=0.0,
        imbalance_price_upper_eur_mwh=0.0,
        mfrr_up_price_lower_eur_mwh=0.0,
        mfrr_up_price_median_eur_mwh=0.0,
        mfrr_up_price_upper_eur_mwh=0.0,
        mfrr_down_price_lower_eur_mwh=0.0,
        mfrr_down_price_median_eur_mwh=0.0,
        mfrr_down_price_upper_eur_mwh=0.0,
        source=ForecastSource(kind="unavailable", window_start=_iso_z(timestamp), method=reason),
    )


def required_data_error(prices: pd.DataFrame, zone: str, timestamp: datetime) -> str | None:
    if prices.empty:
        return "missing_prices"
    cutoff = pd.Timestamp(timestamp)
    recent = prices[(prices["zone"] == zone) & (prices["timestamp_utc"] <= cutoff)]
    if recent.empty:
        return "missing_recent_prices"
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _latest_numeric(frame: pd.DataFrame, column: str, *, default: float) -> float:
    if frame.empty:
        return default
    values = pd.to_numeric(frame.sort_values("timestamp_utc")[column], errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.iloc[-1])


def _zoo_history(prices: pd.DataFrame, loads: pd.DataFrame, issued_at: datetime, zone: str, backend: str) -> np.ndarray:
    pivot = prices[
        (prices["zone"] == zone)
        & (prices["timestamp_utc"] <= pd.Timestamp(issued_at))
    ].pivot_table(index="timestamp_utc", columns="price_type", values="price_eur_mwh", aggfunc="last").sort_index()
    imbalance = pivot.get("imbalance", pd.Series(dtype=float)).ffill()
    day_ahead = pivot.get("day_ahead", imbalance).ffill()
    if backend == "f8":
        load_series = loads[
            (loads["zone"] == zone)
            & (loads["kind"] == "actual")
            & (loads["timestamp_utc"] <= pd.Timestamp(issued_at))
        ].sort_values("timestamp_utc").set_index("timestamp_utc")["load_mw"]
        frame = pd.DataFrame({"imbalance": imbalance, "load": load_series, "day_ahead": day_ahead}).ffill().dropna()
        if frame.empty:
            return np.asarray(imbalance.dropna().tail(192).tolist(), dtype=float)
        return frame.tail(192).to_numpy(dtype=float)
    values = imbalance.dropna().tail(192).tolist()
    if not values:
        values = day_ahead.dropna().tail(192).tolist()
    return np.asarray(values, dtype=float)


def _forecast_history(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    piv = prices.pivot_table(index=["timestamp_utc", "zone"], columns="price_type", values="price_eur_mwh", aggfunc="last").reset_index()
    for column in ["day_ahead", "imbalance", "mfrr_up", "mfrr_down"]:
        if column not in piv.columns:
            piv[column] = np.nan
    spot = piv["day_ahead"].ffill().bfill()
    imbalance = piv["imbalance"].fillna(spot)
    up = piv["mfrr_up"].fillna(imbalance)
    down = piv["mfrr_down"].fillna(imbalance)
    return pd.DataFrame(
        {
            "utc_timestamp": piv["timestamp_utc"],
            "zone": piv["zone"],
            "satisfied_demand_mw": 8.0,
            "imbalance_price_eur_mwh": imbalance,
            "spot_price_eur_mwh": spot,
            "mfrr_marginal_price_up_eur_mwh": up,
            "mfrr_marginal_price_down_eur_mwh": down,
        }
    ).dropna()
