from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from packages.simulator import Bid
from packages.simulator.mfrr_engine import (
    CalibratedMFRRPriceModel,
    MFRRClearingEngine,
    backtest_price_model,
)


def _history() -> pd.DataFrame:
    rows = []
    for zone, direction, slope, intercept in [
        ("DK1", "up", 2.0, 8.0),
        ("DK1", "down", 1.5, 6.0),
        ("DK2", "up", 3.0, 5.0),
        ("DK2", "down", 2.5, 7.0),
    ]:
        for index, volume_mwh in enumerate([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]):
            timestamp = pd.Timestamp("2025-03-04T00:00:00Z") + pd.Timedelta(
                minutes=15 * index
            )
            spot = 50.0 if zone == "DK1" else 55.0
            spread = intercept + slope * volume_mwh
            signed_mw = volume_mwh / 0.25
            if direction == "up":
                rows.append(
                    {
                        "utc_timestamp": timestamp,
                        "zone": zone,
                        "satisfied_demand_mw": signed_mw,
                        "imbalance_price_eur_mwh": spot + spread,
                        "spot_price_eur_mwh": spot,
                        "mfrr_marginal_price_up_eur_mwh": spot + spread,
                        "mfrr_marginal_price_down_eur_mwh": spot,
                    }
                )
            else:
                rows.append(
                    {
                        "utc_timestamp": timestamp,
                        "zone": zone,
                        "satisfied_demand_mw": -signed_mw,
                        "imbalance_price_eur_mwh": spot - spread,
                        "spot_price_eur_mwh": spot,
                        "mfrr_marginal_price_up_eur_mwh": spot,
                        "mfrr_marginal_price_down_eur_mwh": spot - spread,
                    }
                )
    return pd.DataFrame(rows)


def test_calibrated_price_model_fits_zone_direction_supply_curves() -> None:
    model = CalibratedMFRRPriceModel.fit(_history(), min_samples=4)

    dk1_up = model.predict(
        zone="DK1",
        direction="up",
        spot_price_eur_mwh=50.0,
        activation_volume_mwh=10.0,
    )
    dk2_down = model.predict(
        zone="DK2",
        direction="down",
        spot_price_eur_mwh=55.0,
        activation_volume_mwh=8.0,
    )

    assert dk1_up.predicted_imbalance_price_eur_mwh == pytest.approx(78.0)
    assert dk1_up.lower_90_eur_mwh <= dk1_up.predicted_imbalance_price_eur_mwh
    assert dk1_up.upper_90_eur_mwh >= dk1_up.predicted_imbalance_price_eur_mwh
    assert dk1_up.uncertainty_90_eur_mwh >= 0.0
    assert dk1_up.quality.zone == "DK1"
    assert dk1_up.quality.direction == "up"
    assert dk1_up.quality.fallback_level == "zone_direction"
    assert dk1_up.quality.r2 >= 0.99
    assert dk2_down.predicted_imbalance_price_eur_mwh == pytest.approx(28.0)
    assert dk2_down.quality.fallback_level == "zone_direction"


def test_price_model_uses_direction_fallback_when_zone_history_is_sparse() -> None:
    history = _history()
    sparse = history[~((history["zone"] == "DK2") & (history["satisfied_demand_mw"] < 0))]
    model = CalibratedMFRRPriceModel.fit(sparse, min_samples=4)

    prediction = model.predict(
        zone="DK2",
        direction="down",
        spot_price_eur_mwh=55.0,
        activation_volume_mwh=8.0,
    )

    assert prediction.quality.fallback_level == "direction"
    assert prediction.quality.sample_count >= 4
    assert prediction.predicted_imbalance_price_eur_mwh < 55.0


def test_clearing_engine_accepts_merit_order_bids_and_returns_agent_safe_trace() -> None:
    model = CalibratedMFRRPriceModel.fit(_history(), min_samples=4)
    engine = MFRRClearingEngine(model)
    market_row = {
        "utc_timestamp": "2025-03-04T02:00:00Z",
        "zone": "DK1",
        "satisfied_demand_mw": 40.0,
        "imbalance_price_eur_mwh": 78.0,
        "spot_price_eur_mwh": 50.0,
        "mfrr_marginal_price_up_eur_mwh": 78.0,
        "mfrr_marginal_price_down_eur_mwh": 50.0,
    }
    accepted_bid = Bid(
        agent_id="agent-a",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T02:00:00Z",
        side="up",
        quantity_mwh=4.0,
        limit_price_eur_mwh=70.0,
        submitted_at_utc="2025-03-04T01:10:00Z",
    )
    expensive_bid = Bid(
        agent_id="agent-b",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T02:00:00Z",
        side="up",
        quantity_mwh=4.0,
        limit_price_eur_mwh=120.0,
        submitted_at_utc="2025-03-04T01:10:00Z",
    )
    wrong_side_bid = Bid(
        agent_id="agent-c",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T02:00:00Z",
        side="down",
        quantity_mwh=2.0,
        limit_price_eur_mwh=40.0,
        submitted_at_utc="2025-03-04T01:10:00Z",
    )

    result = engine.clear(market_row, [expensive_bid, wrong_side_bid, accepted_bid])
    repeat = engine.clear(market_row, [expensive_bid, wrong_side_bid, accepted_bid])

    assert result.zone == "DK1"
    assert result.direction == "up"
    assert result.baseline_activation_volume_mwh == pytest.approx(10.0)
    assert result.accepted_volume_mwh == pytest.approx(4.0)
    assert result.counterfactual_activation_volume_mwh == pytest.approx(6.0)
    assert result.counterfactual_imbalance_price_eur_mwh == pytest.approx(70.0)
    assert result.price_delta_eur_mwh == pytest.approx(-8.0)
    assert [decision.agent_id for decision in result.accepted_bids] == ["agent-a"]
    assert {decision.reason_code for decision in result.rejected_bids} == {
        "limit_price_not_crossed",
        "opposite_activation_direction",
    }
    assert result.model_quality.fallback_level == "zone_direction"
    assert result.result_hash == repeat.result_hash


def test_down_regulation_clearing_moves_price_toward_spot() -> None:
    model = CalibratedMFRRPriceModel.fit(_history(), min_samples=4)
    engine = MFRRClearingEngine(model)
    market_row = {
        "utc_timestamp": datetime(2025, 3, 4, 2, 0, tzinfo=UTC),
        "zone": "DK2",
        "satisfied_demand_mw": -32.0,
        "imbalance_price_eur_mwh": 28.0,
        "spot_price_eur_mwh": 55.0,
        "mfrr_marginal_price_up_eur_mwh": 55.0,
        "mfrr_marginal_price_down_eur_mwh": 28.0,
    }
    bid = Bid(
        agent_id="p2h",
        asset_id="DK2",
        zone="DK2",
        utc_timestamp="2025-03-04T02:00:00Z",
        side="down",
        quantity_mwh=4.0,
        limit_price_eur_mwh=20.0,
        submitted_at_utc="2025-03-04T01:10:00Z",
    )

    result = engine.clear(market_row, [bid])

    assert result.direction == "down"
    assert result.accepted_volume_mwh == pytest.approx(4.0)
    assert result.counterfactual_activation_volume_mwh == pytest.approx(4.0)
    assert result.counterfactual_imbalance_price_eur_mwh == pytest.approx(38.0)
    assert result.price_delta_eur_mwh == pytest.approx(10.0)


def test_clearing_engine_anchors_counterfactual_to_supplied_baseline_price() -> None:
    model = CalibratedMFRRPriceModel.fit(_history(), min_samples=4)
    engine = MFRRClearingEngine(model)
    market_row = {
        "utc_timestamp": "2025-03-04T02:00:00Z",
        "zone": "DK1",
        "satisfied_demand_mw": 40.0,
        "imbalance_price_eur_mwh": 100.0,
        "spot_price_eur_mwh": 50.0,
        "mfrr_marginal_price_up_eur_mwh": 100.0,
        "mfrr_marginal_price_down_eur_mwh": 50.0,
    }
    bid = Bid(
        agent_id="agent-a",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T02:00:00Z",
        side="up",
        quantity_mwh=4.0,
        limit_price_eur_mwh=90.0,
        submitted_at_utc="2025-03-04T01:10:00Z",
    )

    result = engine.clear(market_row, [bid])

    assert result.counterfactual_imbalance_price_eur_mwh == pytest.approx(92.0)
    assert result.price_delta_eur_mwh == pytest.approx(-8.0)


def test_chronological_backtest_reports_historical_validity_for_both_danish_zones() -> None:
    report = backtest_price_model(_history(), train_fraction=0.67, min_samples=2)

    assert set(report.zone_metrics) == {"DK1", "DK2"}
    assert report.total_rows >= 8
    assert report.mae_eur_mwh <= 1e-9
    assert report.max_abs_error_eur_mwh <= 1e-9
    assert report.interval_coverage_90 >= 0.9
    assert report.result_hash
