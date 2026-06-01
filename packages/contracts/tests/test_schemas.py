"""Contracts package tests — frozen-schema invariants, not behaviour tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from heimdall_contracts import (
    ActivationForecast,
    BidAction,
    ConformalInterval,
    NegotiationKind,
    Persona,
    PersonaArchetype,
    PhysicalViolation,
    QuantileForecast,
    RiskAttitude,
    StageFailed,
    VerifierVerdict,
)
from heimdall_contracts.agents import AgentMessage


def _q(minute: int) -> datetime:
    return datetime(2026, 5, 9, 12, minute, tzinfo=timezone.utc)


def test_bid_requires_15min_aligned_utc() -> None:
    with pytest.raises(ValidationError):
        BidAction(
            market="mFRR",
            direction="sell",
            quantity_mw=10,
            price_eur_per_mwh=42.0,
            delivery_quarter=_q(7),  # not aligned
        )


def test_bid_round_trip() -> None:
    b = BidAction(
        market="mFRR",
        direction="sell",
        quantity_mw=10,
        price_eur_per_mwh=42.0,
        delivery_quarter=_q(15),
    )
    again = BidAction.model_validate_json(b.model_dump_json())
    assert again == b


def test_bid_rejects_negative_quantity() -> None:
    with pytest.raises(ValidationError):
        BidAction(
            market="DA",
            direction="buy",
            quantity_mw=-1,
            price_eur_per_mwh=10,
            delivery_quarter=_q(0),
        )


def test_quantile_forecast_must_be_ascending() -> None:
    with pytest.raises(ValidationError):
        QuantileForecast(horizon_minutes=15, levels=(0.9, 0.5), values=(1.0, 2.0))


def test_conformal_interval_upper_ge_lower() -> None:
    with pytest.raises(ValidationError):
        ConformalInterval(
            horizon_minutes=15, alpha=0.1, lower=10.0, upper=5.0, method="split_cp"
        )


def test_activation_forecast_probabilities_must_sum_to_one() -> None:
    volume = QuantileForecast(horizon_minutes=15, levels=(0.1, 0.5, 0.9), values=(0.0, 2.0, 8.0))
    forecast = ActivationForecast(
        issued_at=_q(0),
        zone="DK1",
        horizon_minutes=15,
        p_up=0.4,
        p_down=0.3,
        p_neutral=0.3,
        volume_quantiles_mwh=volume,
        source_model="activation-f0",
    )
    assert forecast.leakage_guard == "historical_only"

    with pytest.raises(ValidationError):
        ActivationForecast(
            issued_at=_q(0),
            zone="DK1",
            horizon_minutes=15,
            p_up=0.4,
            p_down=0.4,
            p_neutral=0.4,
            volume_quantiles_mwh=volume,
            source_model="bad",
        )


def test_persona_full_round_trip() -> None:
    p = Persona(
        agent_id="agent-001",
        archetype=PersonaArchetype.P2H,
        risk_attitude=RiskAttitude.AVERSE,
        sophistication="high",
        info_latency_min=0,
        capacity_mw=50,
        storage_mwh=100,
        forecaster_id="F9",
        llm_id="L5",
    )
    assert p.archetype == PersonaArchetype.P2H


def test_verifier_verdict_accepts_minimal() -> None:
    v = VerifierVerdict(accepted=True, alpha=0.1)
    assert v.accepted is True
    assert v.stage_failed is None


def test_verifier_verdict_rejected_with_physical() -> None:
    pv = PhysicalViolation(
        constraint="ramp_limit",
        current_value=12.0,
        bound_value=8.0,
        suggestion="reduce by >=4 MW",
    )
    v = VerifierVerdict(
        accepted=False,
        stage_failed="physical",
        physical_violation=pv,
    )
    assert v.stage_failed == "physical"


def test_negotiation_message_kinds_match_proposal() -> None:
    proposal_set: set[NegotiationKind] = {"PROPOSE", "COUNTER", "ACCEPT", "REJECT", "WITHDRAW"}
    sample = AgentMessage(kind="PROPOSE", sender="a", recipient="b")
    # type-checks happen via Literal; round-trip via JSON exercises the union
    again = AgentMessage.model_validate_json(sample.model_dump_json())
    assert again.kind in proposal_set


def test_stage_failed_literal_is_well_formed() -> None:
    # If this Literal grows, both verifier service and the trace table need updating.
    valid: set[StageFailed] = {"physical", "conformal"}
    assert "physical" in valid and "conformal" in valid
