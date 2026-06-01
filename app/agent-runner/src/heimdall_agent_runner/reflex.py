"""Reflex baseline policy. Stub for day 1.

The reflex agent emits a small, risk-neutral sell-the-headroom bid against the
last observed price. It deliberately *does not* call any LLM. This keeps day-1
end-to-end smoke tests (`market-simulator` -> `agent-runner` -> `verifier`)
runnable on CPU while the B200 is busy.

TODO(P1, sprint days 5-6): replace `decide` with a vLLM-backed call to
`Qwen3.6-35B-A3B` per docs/RESEARCH-PROPOSAL.md §4.2.2 / §4.7. Persona prompt
fragments live in `heimdall_personas`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from heimdall_contracts import BidAction, MarketState, Persona


class AgentDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona: Persona
    market_state: MarketState
    target_market: Literal["DA", "ID", "mFRR"] = "mFRR"


class AgentDecisionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bid: BidAction | None
    rationale: str
    used_llm: bool = Field(default=False, description="False on day 1 (reflex baseline).")


def decide(req: AgentDecisionRequest) -> AgentDecisionResponse:
    state = req.market_state
    last_price = (
        state.last_mfrr_price_eur_per_mwh
        or state.last_da_price_eur_per_mwh
        or state.last_id_price_eur_per_mwh
    )
    if last_price is None:
        return AgentDecisionResponse(
            bid=None,
            rationale="no last price available; abstaining (reflex baseline)",
        )

    next_quarter = _next_15min_after(state.timestamp)
    qty = min(req.persona.capacity_mw * 0.1, 5.0)
    direction: Literal["buy", "sell"] = "sell"  # reflex: always sell at +1% over last
    bid = BidAction(
        market=req.target_market,
        direction=direction,
        quantity_mw=qty,
        price_eur_per_mwh=round(last_price * 1.01, 2),
        delivery_quarter=next_quarter,
    )
    return AgentDecisionResponse(
        bid=bid,
        rationale="reflex: sell 10% of capacity at +1% over last observed price",
    )


def _next_15min_after(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    minute_floor = (ts.minute // 15) * 15
    floored = ts.replace(minute=minute_floor, second=0, microsecond=0)
    return floored + timedelta(minutes=15)


# --- FastAPI surface --------------------------------------------------------

app = FastAPI(title="heimdall-agent-runner", version="0.0.1")


@app.post("/decide", response_model=AgentDecisionResponse)
def decide_route(req: AgentDecisionRequest) -> AgentDecisionResponse:
    return decide(req)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
