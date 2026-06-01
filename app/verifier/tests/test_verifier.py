"""Verifier tests covering physical + conformal stages and the orchestrator."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from heimdall_contracts import BidAction, ConformalInterval
from heimdall_verifier.conformal import conformal_check
from heimdall_verifier.physical import (
    AssetSpec,
    default_p2h_spec,
    default_p2h_state,
    physical_check,
)
from heimdall_verifier.service import VerifyRequest, app, verify


def _bid(direction: str, qty: float, price: float = 50.0, *, market: str = "mFRR") -> BidAction:
    return BidAction(
        market=market,  # type: ignore[arg-type]
        direction=direction,  # type: ignore[arg-type]
        quantity_mw=qty,
        price_eur_per_mwh=price,
        delivery_quarter=datetime(2026, 5, 9, 12, 30, tzinfo=timezone.utc),
    )


# --- Physical-stage unit tests ---------------------------------------------


def test_physical_accepts_in_envelope_bid() -> None:
    assert physical_check(_bid("sell", 5.0), default_p2h_spec(), default_p2h_state()) is None


def test_physical_rejects_over_capacity_bid() -> None:
    pv = physical_check(_bid("sell", 60.0), default_p2h_spec(), default_p2h_state())
    assert pv is not None and pv.constraint == "position_envelope"


def test_physical_rejects_ramp_violation() -> None:
    spec = AssetSpec(q_max_mw=50.0, ramp_mw_per_min=0.1, storage_mwh=100.0, cop=3.0)
    pv = physical_check(_bid("sell", 10.0), spec, default_p2h_state())
    assert pv is not None and pv.constraint == "ramp_limit"


def test_physical_rejects_soc_floor_for_oversell() -> None:
    state = replace(default_p2h_state(), soc_mwh=1.0)
    pv = physical_check(_bid("sell", 30.0), default_p2h_spec(), state)
    assert pv is not None and pv.constraint == "soc_floor"


def test_physical_rejects_soc_ceiling_for_overbuy() -> None:
    state = replace(default_p2h_state(), soc_mwh=99.0)
    pv = physical_check(_bid("buy", 30.0), default_p2h_spec(), state)
    assert pv is not None and pv.constraint == "soc_ceiling"


def test_physical_rejects_gate_closure_passed() -> None:
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    state = replace(
        default_p2h_state(now=now),
        gate_closure_utc=now + timedelta(hours=1),
    )
    pv = physical_check(_bid("sell", 5.0), default_p2h_spec(), state)
    assert pv is not None and pv.constraint == "gate_closure"


def test_physical_rejects_offtick_price() -> None:
    bid = _bid("sell", 5.0, price=50.005)  # default tick is 0.01
    pv = physical_check(bid, default_p2h_spec(), default_p2h_state())
    assert pv is not None and pv.constraint == "bid_tick_size"


def test_physical_rejects_cash_floor() -> None:
    state = replace(default_p2h_state(), cash_eur=10.0)
    pv = physical_check(_bid("buy", 10.0, price=50.0), default_p2h_spec(), state)
    assert pv is not None and pv.constraint == "cash_floor"


# --- Conformal-stage unit tests --------------------------------------------


def _interval(lo: float, hi: float, *, alpha: float = 0.1) -> ConformalInterval:
    return ConformalInterval(
        horizon_minutes=15, alpha=alpha, lower=lo, upper=hi, method="aci"
    )


def test_conformal_accepts_when_worst_case_profit_above_tau() -> None:
    bid = _bid("sell", 10.0, price=40.0)
    # Sell @ 40, interval [50, 80] -> always clears -> worst case at p=50:
    # profit = (50 - 40) * 10 * 0.25 = +25 EUR  >  tau (-100)
    accepted, pi_min = conformal_check(bid, _interval(50.0, 80.0), tau_eur=-100.0)
    assert accepted is True
    assert pi_min == pytest.approx(25.0)


def test_conformal_rejects_when_worst_case_profit_below_tau() -> None:
    bid = _bid("buy", 10.0, price=100.0)
    # Buy @ 100, interval [60, 100] -> at upper p=100, fill = qty, profit = 0;
    # at lower p=60 profit = (100-60)*10*0.25 = +100. min = 0.
    # Make tau strict: tau = +50 -> reject.
    accepted, pi_min = conformal_check(bid, _interval(60.0, 100.0), tau_eur=50.0)
    assert accepted is False
    assert pi_min == 0.0


# --- Service / orchestration ----------------------------------------------


def _verify_req(bid: BidAction, *, lo: float, hi: float) -> VerifyRequest:
    spec = default_p2h_spec()
    state = default_p2h_state()
    return VerifyRequest(
        bid=bid,
        spec={
            "q_max_mw": spec.q_max_mw,
            "ramp_mw_per_min": spec.ramp_mw_per_min,
            "storage_mwh": spec.storage_mwh,
            "cop": spec.cop,
            "loss_per_quarter": spec.loss_per_quarter,
            "bid_tick_eur": spec.bid_tick_eur,
        },  # type: ignore[arg-type]
        state={
            "position_mw": state.position_mw,
            "last_delta_mw": state.last_delta_mw,
            "soc_mwh": state.soc_mwh,
            "cash_eur": state.cash_eur,
            "now_utc": state.now_utc,
            "gate_closure_utc": state.gate_closure_utc,
        },  # type: ignore[arg-type]
        interval=_interval(lo, hi),
        tau_eur=-100.0,
    )


def test_service_accepts_clean_bid() -> None:
    v = verify(_verify_req(_bid("sell", 5.0, price=40.0), lo=50.0, hi=80.0))
    assert v.accepted is True
    assert v.stage_failed is None


def test_service_rejects_capacity_violation_at_physical_stage() -> None:
    v = verify(_verify_req(_bid("sell", 60.0, price=40.0), lo=50.0, hi=80.0))
    assert v.accepted is False
    assert v.stage_failed == "physical"
    assert v.physical_violation is not None
    assert v.physical_violation.constraint == "position_envelope"


def test_service_rejects_negative_worst_case_profit_at_alpha_0_1() -> None:
    # Sell @ 40, interval [10, 30] -> clears nowhere -> profit = 0 always.
    # Pick tau = +1 -> reject.
    bid = _bid("sell", 10.0, price=40.0)
    req = _verify_req(bid, lo=10.0, hi=30.0)
    req = req.model_copy(update={"tau_eur": 1.0})
    v = verify(req)
    assert v.accepted is False
    assert v.stage_failed == "conformal"
    assert v.worst_case_profit_eur == 0.0


def test_healthz_route() -> None:
    client = TestClient(app)
    assert client.get("/healthz").json() == {"status": "ok"}


def test_verify_route_round_trips_through_fastapi() -> None:
    client = TestClient(app)
    req = _verify_req(_bid("sell", 5.0, price=40.0), lo=50.0, hi=80.0)
    resp = client.post("/verify", json=req.model_dump(mode="json"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] is True
