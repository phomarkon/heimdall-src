from __future__ import annotations

import pandas as pd
import pytest

from packages.simulator.forecast import (
    BaselineMFRRForecaster,
    ForecastMarketState,
    ForecastSource,
)


def _history() -> pd.DataFrame:
    rows = []
    for zone in ["DK1", "DK2"]:
        base = 50.0 if zone == "DK1" else 60.0
        for day in range(8):
            for tick in range(4):
                volume_mwh = 4.0 + tick
                timestamp = pd.Timestamp("2025-03-04T00:00:00Z") + pd.Timedelta(
                    days=day, minutes=15 * tick
                )
                rows.append(
                    {
                        "utc_timestamp": timestamp,
                        "zone": zone,
                        "satisfied_demand_mw": volume_mwh / 0.25,
                        "imbalance_price_eur_mwh": base + 10.0 + volume_mwh,
                        "spot_price_eur_mwh": base,
                        "mfrr_marginal_price_up_eur_mwh": base + 10.0 + volume_mwh,
                        "mfrr_marginal_price_down_eur_mwh": base - 10.0,
                    }
                )
    return pd.DataFrame(rows)


def test_forecast_market_state_converts_to_market_row_without_oracle_labels() -> None:
    state = ForecastMarketState(
        delivery_timestamp="2025-03-12T00:45:00Z",
        zone="DK1",
        issued_at="2025-03-11T23:00:00Z",
        activation_direction="up",
        activation_volume_mwh=5.0,
        spot_price_eur_mwh=50.0,
        imbalance_price_lower_eur_mwh=60.0,
        imbalance_price_median_eur_mwh=66.0,
        imbalance_price_upper_eur_mwh=80.0,
        mfrr_up_price_lower_eur_mwh=60.0,
        mfrr_up_price_median_eur_mwh=66.0,
        mfrr_up_price_upper_eur_mwh=80.0,
        mfrr_down_price_lower_eur_mwh=40.0,
        mfrr_down_price_median_eur_mwh=45.0,
        mfrr_down_price_upper_eur_mwh=50.0,
        source=ForecastSource(kind="baseline", window_start="2025-03-04T00:00:00Z"),
    )

    row = state.to_market_row()

    assert row["utc_timestamp"] == "2025-03-12T00:45:00Z"
    assert row["zone"] == "DK1"
    assert row["satisfied_demand_mw"] == pytest.approx(20.0)
    assert row["imbalance_price_eur_mwh"] == pytest.approx(66.0)
    assert row["mfrr_marginal_price_up_eur_mwh"] == pytest.approx(66.0)
    assert state.interval_for_side("up") == (60.0, 80.0)


def test_baseline_forecaster_emits_deterministic_conformal_forecasts_for_both_zones() -> None:
    forecaster = BaselineMFRRForecaster.fit(
        _history(),
        lookback_days=7,
        calibration_fraction=0.25,
        alpha=0.1,
    )

    first = forecaster.forecast(
        delivery_timestamp="2025-03-12T00:45:00Z",
        zone="DK1",
        issued_at="2025-03-11T23:00:00Z",
    )
    second = forecaster.forecast(
        delivery_timestamp="2025-03-12T00:45:00Z",
        zone="DK1",
        issued_at="2025-03-11T23:00:00Z",
    )
    dk2 = forecaster.forecast(
        delivery_timestamp="2025-03-12T00:45:00Z",
        zone="DK2",
        issued_at="2025-03-11T23:00:00Z",
    )

    assert first.zone == "DK1"
    assert dk2.zone == "DK2"
    assert first.activation_direction == "up"
    assert first.imbalance_price_lower_eur_mwh <= first.imbalance_price_median_eur_mwh
    assert first.imbalance_price_upper_eur_mwh >= first.imbalance_price_median_eur_mwh
    assert first.result_hash == second.result_hash
    assert first.source.kind == "baseline_conformal"


def test_baseline_forecaster_raises_structured_error_for_missing_zone() -> None:
    forecaster = BaselineMFRRForecaster.fit(_history(), lookback_days=7)

    with pytest.raises(ValueError, match="No forecast history"):
        forecaster.forecast(
            delivery_timestamp="2025-03-12T00:45:00Z",
            zone="NO2",
            issued_at="2025-03-11T23:00:00Z",
        )


def test_baseline_forecaster_does_not_use_rows_after_issue_time() -> None:
    history = pd.concat(
        [
            _history(),
            pd.DataFrame(
                [
                    {
                        "utc_timestamp": "2025-03-11T23:45:00Z",
                        "zone": "DK1",
                        "satisfied_demand_mw": 400.0,
                        "imbalance_price_eur_mwh": 9999.0,
                        "spot_price_eur_mwh": 9999.0,
                        "mfrr_marginal_price_up_eur_mwh": 9999.0,
                        "mfrr_marginal_price_down_eur_mwh": 9999.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    forecaster = BaselineMFRRForecaster.fit(history, lookback_days=7)

    forecast = forecaster.forecast(
        delivery_timestamp="2025-03-12T00:45:00Z",
        zone="DK1",
        issued_at="2025-03-11T23:00:00Z",
    )

    assert forecast.imbalance_price_upper_eur_mwh < 9999.0
