"""Tests for the data-grounded delivery-shortfall downside in tools/evaluation/rescore_runs.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluation.rescore_runs import (
    _availability_by_archetype,
    _per_run_metrics,
    _realized_availability,
    _shortfall_settlement,
)


def _rho(cv: float, a_star: float, *, agent: str = "agent-0", side: str = "up") -> float:
    return _realized_availability(
        run_seed=42, ts="2026-04-01T00:00:00Z", zone="DK1", side=side,
        agent_id=agent, archetype="renewables", cv=cv, a_star=a_star,
    )


def test_cv_zero_recovers_expected_availability() -> None:
    assert _rho(0.0, 0.55) == 0.55  # no draw -> status quo


def test_firm_asset_has_no_delivery_risk() -> None:
    assert _rho(0.25, 1.0) == 1.0
    sf, penalty = _shortfall_settlement(side="up", cleared=4.0, a_star=1.0, rho=1.0,
                                        settlement=80.0, imbalance=130.0)
    assert sf == 0.0 and penalty == 0.0


def test_no_shortfall_when_realized_meets_expectation() -> None:
    sf, penalty = _shortfall_settlement(side="up", cleared=4.0, a_star=0.55, rho=0.55,
                                        settlement=80.0, imbalance=130.0)
    assert sf == 0.0 and penalty == 0.0


def test_up_shortfall_is_a_loss_when_imbalance_above_settlement() -> None:
    # rho = a*/2 -> deliver half -> shortfall = 2.0 MWh; adverse gap = I - S = 50.
    sf, penalty = _shortfall_settlement(side="up", cleared=4.0, a_star=0.50, rho=0.25,
                                        settlement=80.0, imbalance=130.0)
    assert sf == 2.0
    assert penalty == 2.0 * (130.0 - 80.0)  # +100 loss


def test_up_shortfall_is_a_windfall_when_imbalance_below_settlement() -> None:
    sf, penalty = _shortfall_settlement(side="up", cleared=4.0, a_star=0.50, rho=0.25,
                                        settlement=80.0, imbalance=30.0)
    assert sf == 2.0
    assert penalty == 2.0 * (30.0 - 80.0)  # -100 windfall


def test_down_shortfall_sign_convention() -> None:
    # DOWN loss when settlement > imbalance (adverse gap = S - I).
    sf, penalty = _shortfall_settlement(side="down", cleared=4.0, a_star=0.50, rho=0.25,
                                        settlement=80.0, imbalance=30.0)
    assert sf == 2.0
    assert penalty == 2.0 * (80.0 - 30.0)  # +100 loss


def test_draw_is_deterministic_and_condition_label_independent() -> None:
    # Identical physical-bid identity -> identical draw across repeated/"different-condition" calls.
    assert _rho(0.2, 0.55, agent="agent-3") == _rho(0.2, 0.55, agent="agent-3")
    # A different bid (different agent) gets its own draw.
    assert _rho(0.2, 0.55, agent="agent-3") != _rho(0.2, 0.55, agent="agent-9")


def test_draw_is_mean_preserving() -> None:
    a_star = 0.55
    draws = [
        _realized_availability(run_seed=42, ts=f"t{i}", zone="DK1", side="up",
                               agent_id=f"a{i}", archetype="renewables", cv=0.15, a_star=a_star)
        for i in range(4000)
    ]
    assert abs(float(np.mean(draws)) - a_star) < 0.01


def test_availability_extracted_from_nested_trace_result(tmp_path: Path) -> None:
    line = {
        "archetype": "renewables",
        "tool_calls": [
            {"name": "simulate_renewables_bid",
             "result": {"projected_state": {"availability_share": 0.55}}},
            {"name": "simulate_renewables_bid",
             "result": {"projected_state": {"availability_share": 0.55}}},
        ],
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(line) + "\n")
    assert _availability_by_archetype(path) == {"renewables": 0.55}


def test_per_run_cv_zero_equals_realized() -> None:
    ts = pd.Timestamp("2026-04-01T00:00:00Z")
    bids = pd.DataFrame([
        {"timestamp_utc": ts, "zone": "DK1", "agent_id": "a0", "archetype": "renewables",
         "side": "up", "quantity_mwh": 4.0, "status": "filled", "cleared_mwh": 4.0,
         "realized_profit_eur": 120.0},
    ])
    truth = pd.DataFrame([
        {"timestamp_utc": ts, "zone": "DK1", "activation_direction": "up",
         "activated_volume_mwh": 100.0, "spot_price_eur_mwh": 50.0,
         "settlement_price_eur_mwh": 80.0, "imbalance_price_eur_mwh": 130.0},
    ])
    out = _per_run_metrics(bids, truth, lambdas=[], cvs=[0.0, 0.25],
                           a_star_by_archetype={"renewables": 0.55}, run_seed=42)
    assert out["delivery_adjusted_profit_eur_cv0.0"] == out["realized_profit_eur"]
    assert out["shortfall_loss_eur_cv0.25"] >= 0.0
