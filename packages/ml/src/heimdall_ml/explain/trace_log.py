"""Interpretability log format. Consumed by the paper's figure pipeline (Fig. 1
hero figure + per-decision audit trail) and by the frontend's trace inspector.

Single record per (run_id, step, agent_id, decision_id). Append-only JSONL.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExplanationRecord(BaseModel):
    """One auditable explanation record. Emitted by the focal verifier and by
    every downstream XAI tool that touches the focal-agent path."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    step: int = Field(ge=0)
    agent_id: str
    decision_id: str
    kind: Literal["forecast_shap", "verifier_attribution", "rationale_quality"]
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict
