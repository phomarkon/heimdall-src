from __future__ import annotations

from dataclasses import replace

from heimdall_ai_society.runner import _matching_accepted_simulation
from heimdall_ai_society.schemas import LLMBidDecision, ToolCallRecord
from heimdall_ai_society.tools import AgentToolExecutor
from heimdall_contracts import PersonaArchetype
from heimdall_personas import default_persona

from packages.simulator import (
    Bid,
    ForecastMarketState,
    ForecastSource,
    RealAssetSpec,
    simulate_real_asset_bid,
)


def _forecast() -> ForecastMarketState:
    return ForecastMarketState(
        delivery_timestamp="2025-03-04T03:00:00Z",
        zone="DK1",
        issued_at="2025-03-04T02:00:00Z",
        activation_direction="up",
        activation_volume_mwh=5.0,
        spot_price_eur_mwh=50.0,
        imbalance_price_lower_eur_mwh=90.0,
        imbalance_price_median_eur_mwh=100.0,
        imbalance_price_upper_eur_mwh=110.0,
        mfrr_up_price_lower_eur_mwh=90.0,
        mfrr_up_price_median_eur_mwh=100.0,
        mfrr_up_price_upper_eur_mwh=110.0,
        mfrr_down_price_lower_eur_mwh=20.0,
        mfrr_down_price_median_eur_mwh=30.0,
        mfrr_down_price_upper_eur_mwh=40.0,
        source=ForecastSource(kind="baseline", window_start="2025-03-04T00:00:00Z"),
    )


def _bid(*, side: str = "up", quantity_mwh: float = 1.0) -> Bid:
    return Bid(
        agent_id="agent",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T03:00:00Z",
        side=side,
        quantity_mwh=quantity_mwh,
        limit_price_eur_mwh=80.0,
    )


def test_p2h_market_proxy_accepts_positive_opportunity_and_hashes_deterministically() -> None:
    persona = default_persona("agent-p2h", PersonaArchetype.P2H)
    executor = AgentToolExecutor(
        persona=persona,
        forecast=_forecast(),
        data_tools=None,
        simulator_tool=None,
        asset_simulator_mode="proxy",
    )
    args = {"side": "up", "quantity_mwh": 0.25, "limit_price_eur_mwh": 80.0}
    first = executor.execute("simulate_bid", args)
    second = executor.execute("simulate_bid", args)
    assert first.ok is True
    assert first.result["backend"] == "proxy"
    assert first.result["authority"] == "proxy_comparison"
    assert first.result["controls_acceptance"] is True
    assert first.result["accepted"] is True
    assert first.result["expected_profit_eur"] > 0
    assert first.result["worst_case_profit_eur"] >= 0
    assert first.result["physical_limit_mwh"] >= args["quantity_mwh"]
    assert first.result["result_hash"] == second.result["result_hash"]


def test_p2h_market_proxy_rejects_negative_worst_case_spread() -> None:
    persona = default_persona("agent-p2h", PersonaArchetype.P2H)
    forecast = replace(_forecast(), spot_price_eur_mwh=120.0)
    executor = AgentToolExecutor(
        persona=persona,
        forecast=forecast,
        data_tools=None,
        simulator_tool=None,
        asset_simulator_mode="proxy",
    )
    record = executor.execute(
        "simulate_bid",
        {"side": "up", "quantity_mwh": 0.25, "limit_price_eur_mwh": 80.0},
    )
    assert record.ok is True
    assert record.result["accepted"] is False
    assert "negative_worst_case_spread" in record.result["reason_codes"]


def test_ev_market_proxy_is_less_physical_than_asset_light_proxy() -> None:
    persona = default_persona("agent-ev", PersonaArchetype.EV)

    class _ActivationMismatchTools:
        def get_activation_context(self, **_: object) -> dict[str, object]:
            return {"direction_hint": "down"}

    args = {"side": "up", "quantity_mwh": 0.25, "limit_price_eur_mwh": 80.0}
    market_proxy = AgentToolExecutor(
        persona=persona,
        forecast=_forecast(),
        data_tools=_ActivationMismatchTools(),
        simulator_tool=None,
        asset_simulator_mode="proxy",
        asset_proxy_style="market",
    ).execute("simulate_ev_bid", args)
    asset_light_proxy = AgentToolExecutor(
        persona=persona,
        forecast=_forecast(),
        data_tools=_ActivationMismatchTools(),
        simulator_tool=None,
        asset_simulator_mode="proxy",
        asset_proxy_style="asset_light",
    ).execute("simulate_ev_bid", args)

    assert market_proxy.result["backend"] == "proxy"
    assert market_proxy.result["simulator_kind"] == "ev_proxy_mfrr"
    assert market_proxy.result["accepted"] is True
    assert asset_light_proxy.result["simulator_kind"] == "ev_virtual_battery"
    assert asset_light_proxy.result["accepted"] is False
    assert "activation_prior_side_mismatch" in asset_light_proxy.result["reason_codes"]


def test_generator_real_asset_accepts_feasible_bid_and_hashes_deterministically() -> None:
    spec = RealAssetSpec(
        archetype="generator",
        capacity_mw=200.0,
        current_dispatch_share=0.5,
        ramp_share_per_tick=0.35,
        marginal_cost_eur_mwh=5.0,
    )
    first = simulate_real_asset_bid(spec=spec, bid=_bid(quantity_mwh=2.0), forecast=_forecast(), tau_eur=0.0)
    second = simulate_real_asset_bid(spec=spec, bid=_bid(quantity_mwh=2.0), forecast=_forecast(), tau_eur=0.0)
    assert first.accepted is True
    assert first.backend == "scenario_envelope"
    assert first.authority == "authoritative"
    assert first.next_state is not None
    assert first.result_hash == second.result_hash


def test_wind_real_asset_rejects_impossible_down_bid() -> None:
    result = simulate_real_asset_bid(
        spec=RealAssetSpec(archetype="wind", capacity_mw=50.0, availability_share=0.45),
        bid=_bid(side="down", quantity_mwh=1.0),
        forecast=_forecast(),
        tau_eur=-100.0,
    )
    assert result.accepted is False
    assert "wind_down_bid_not_physically_supported" in result.reason_codes


def test_final_guard_only_accepts_top_level_controlling_backend() -> None:
    decision = LLMBidDecision(
        action="bid",
        side="up",
        quantity_mwh=2.0,
        limit_price_eur_mwh=80.0,
        rationale="test",
        confidence=0.5,
    )
    args = {"side": "up", "quantity_mwh": 2.0, "limit_price_eur_mwh": 80.0}
    proxy_only_inside_comparison = ToolCallRecord(
        name="simulate_generator_bid",
        arguments=args,
        ok=True,
        result={
            "accepted": False,
            "backend": "pypsa_background",
            "authority": "authoritative",
            "controls_acceptance": True,
            "comparison": {"proxy": {"accepted": True}},
        },
    )
    controlling_pypsa = ToolCallRecord(
        name="simulate_generator_bid",
        arguments=args,
        ok=True,
        result={
            "accepted": True,
            "backend": "pypsa_background",
            "authority": "authoritative",
            "controls_acceptance": True,
        },
    )

    assert _matching_accepted_simulation("simulate_generator_bid", decision, [proxy_only_inside_comparison]) is None
    assert _matching_accepted_simulation("simulate_generator_bid", decision, [controlling_pypsa]) is controlling_pypsa
