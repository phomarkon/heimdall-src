"""Agent / persona / negotiation message schemas. Per docs/RESEARCH-PROPOSAL.md
§4.2.1 (information-access matrix), §4.2.2 (model zoo), §4.2.3 (negotiation),
§6.3 (trace table)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from heimdall_contracts.bids import BidAction
from heimdall_contracts.verifier import VerifierVerdict


class PersonaArchetype(str, Enum):
    WIND = "wind"
    EV = "ev"
    RETAILER = "retailer"
    P2H = "p2h"
    GENERATOR = "generator"
    RENEWABLES = "renewables"
    ARBITRAGEUR = "arbitrageur"
    MARKET_MECHANICS_EXPERT = "market_mechanics_expert"
    IMBALANCE_ANALYTICS_EXPERT = "imbalance_analytics_expert"
    TRADING_RISK_MONITOR = "trading_risk_monitor"
    GRID_CONSTRAINT_ANALYST = "grid_constraint_analyst"
    OUTAGE_IMPACT_SCORER = "outage_impact_scorer"
    LIMIT_PRICE_SPECIALIST = "limit_price_specialist"
    CANDIDATE_SIZING_SPECIALIST = "candidate_sizing_specialist"
    UNCERTAINTY_AUDITOR = "uncertainty_auditor"
    DECISION_AUDITOR = "decision_auditor"


class RiskAttitude(str, Enum):
    AVERSE = "averse"
    NEUTRAL = "neutral"
    SEEKING = "seeking"


class Persona(BaseModel):
    """One agent's static configuration. Six diversity axes (§4.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    archetype: PersonaArchetype
    risk_attitude: RiskAttitude
    sophistication: Literal["low", "medium", "high"]
    info_latency_min: int = Field(ge=0)
    capacity_mw: float = Field(ge=0.0)
    storage_mwh: float | None = Field(default=None, ge=0.0)
    forecaster_id: str = Field(description="F-code from the forecaster zoo (e.g. 'F9').")
    llm_id: str = Field(description="L-code from the LLM zoo (e.g. 'L5').")


NegotiationKind = Literal["PROPOSE", "COUNTER", "ACCEPT", "REJECT", "WITHDRAW"]


class AgentMessage(BaseModel):
    """Bilateral negotiation message (§4.2.3). Five typed kinds; not free-form chat."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: NegotiationKind
    sender: str
    recipient: str
    market: Literal["DA", "ID", "mFRR"] | None = None
    quantity_mw: float | None = Field(default=None, ge=0.0)
    price_eur_per_mwh: float | None = None
    expires_at: datetime | None = None
    in_reply_to: str | None = None
    proposal_id: str | None = None
    reason: str | None = None


class AgentTrace(BaseModel):
    """The single source of truth for the trace hypertable (§6.3)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    step: int = Field(ge=0)
    timestamp: datetime
    agent_id: str
    persona: Persona
    state: dict
    reasoning: str
    tool_calls: list[dict]
    proposed_action: BidAction | None = None
    verifier_verdict: VerifierVerdict | None = None
    realized_outcome: dict | None = None
