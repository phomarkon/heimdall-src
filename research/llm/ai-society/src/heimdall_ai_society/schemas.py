from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

WatchLabel = Literal["must_watch", "watch", "ignore"]
RiskLabel = Literal["low", "medium", "high"]
OpportunityLabel = Literal["none", "weak", "actionable"]
PriorityLabel = Literal["low", "medium", "high", "critical"]
OperatorAction = Literal["ignore", "monitor", "inspect", "prepare_bid", "escalate"]
ToolCallProvenance = Literal[
    "runner_seeded",
    "llm_requested",
    "forced_final",
    "runner_diagnostic",
    "retry",
    "unknown",
]
PriorityReason = Literal[
    "none",
    "activation",
    "profit_edge",
    "accepted_candidate",
    "uncertainty",
    "rejection_cluster",
    "cross_agent_disagreement",
]
WatchReason = Literal[
    "activation_risk",
    "price_volatility",
    "forecast_uncertainty",
    "accepted_bid_available",
    "verifier_rejection_cluster",
    "cross_agent_disagreement",
]


class LLMBidDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["bid", "watch", "abstain"]
    side: Literal["up", "down"] | None = None
    quantity_mwh: float | None = Field(default=None, gt=0.0)
    limit_price_eur_mwh: float | None = None
    rationale: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    watch_label: WatchLabel = "ignore"
    risk_label: RiskLabel = "low"
    uncertainty_label: RiskLabel = "low"
    opportunity_label: OpportunityLabel = "none"
    watch_reasons: list[WatchReason] = Field(default_factory=list)
    priority_label: PriorityLabel = "low"
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    operator_action: OperatorAction = "ignore"
    priority_reason: PriorityReason = "none"

    @field_validator("watch_reasons")
    @classmethod
    def _dedupe_watch_reasons(cls, value: list[WatchReason]) -> list[WatchReason]:
        return list(dict.fromkeys(value))


class DeliberationNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    side_belief: Literal["up", "down", "mixed", "none"]
    price_belief: str = Field(min_length=1, max_length=240)
    uncertainty_label: RiskLabel = "medium"
    risk_concern: str = Field(default="", max_length=240)
    requested_peer_id: str | None = Field(default=None, max_length=80)
    requested_archetype: str | None = Field(default=None, max_length=80)
    requested_tool: str | None = Field(default=None, max_length=80)
    requested_candidate: dict[str, object] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list, max_length=6)
    rationale: str = Field(min_length=1, max_length=600)


class PeerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_agent_id: str | None = Field(default=None, max_length=80)
    agreement: Literal["agree", "object", "uncertain"]
    suggested_side: Literal["up", "down"] | None = None
    suggested_quantity_mwh: float | None = Field(default=None, gt=0.0)
    suggested_limit_price_eur_mwh: float | None = None
    evidence_refs: list[str] = Field(default_factory=list, max_length=6)
    rationale: str = Field(min_length=1, max_length=600)


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict = Field(default_factory=dict)
    ok: bool
    result: dict = Field(default_factory=dict)
    error: str | None = None
    provenance: ToolCallProvenance = "unknown"


class SocietyTraceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    step: int
    timestamp: datetime
    observed_at: datetime
    agent_id: str
    zone: str
    archetype: str
    agent_role: str = "action_agent"
    llm_id: str
    forecaster_id: str
    forecast_backend: str | None = None
    decision: LLMBidDecision
    verifier_mode: str
    verifier_accepted: bool | None = None
    verifier_reason_codes: list[str] = Field(default_factory=list)
    market_price_eur_mwh: float
    forecast_interval_eur_mwh: tuple[float, float]
    rationale: str
    unavailable_reason: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    seeded_tool_call_count: int = 0
    llm_tool_call_count: int = 0
    forced_tool_call_count: int = 0
    diagnostic_tool_call_count: int = 0
    retry_tool_call_count: int = 0
    unknown_tool_call_count: int = 0
    tool_call_provenance_counts: dict[str, int] = Field(default_factory=dict)
    memory_item_count: int = 0
    memory_fingerprint: str | None = None
    memory_lessons: list[dict[str, object]] = Field(default_factory=list)
