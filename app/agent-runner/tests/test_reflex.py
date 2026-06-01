from datetime import datetime, timezone

from heimdall_contracts import MarketState, PersonaArchetype
from heimdall_agent_runner import decide
from heimdall_agent_runner.reflex import AgentDecisionRequest
from heimdall_personas import default_persona


def test_reflex_emits_a_sell_bid_when_price_is_known() -> None:
    persona = default_persona("agent-001", PersonaArchetype.WIND)
    state = MarketState(
        tick=42,
        timestamp=datetime(2026, 5, 9, 12, 7, tzinfo=timezone.utc),
        zone="DK1",
        last_mfrr_price_eur_per_mwh=50.0,
    )
    resp = decide(AgentDecisionRequest(persona=persona, market_state=state))
    assert resp.bid is not None
    assert resp.bid.direction == "sell"
    assert resp.bid.delivery_quarter.minute % 15 == 0
    assert resp.used_llm is False


def test_reflex_abstains_without_a_last_price() -> None:
    persona = default_persona("agent-001", PersonaArchetype.WIND)
    state = MarketState(
        tick=0, timestamp=datetime(2026, 5, 9, tzinfo=timezone.utc), zone="DK1"
    )
    resp = decide(AgentDecisionRequest(persona=persona, market_state=state))
    assert resp.bid is None
