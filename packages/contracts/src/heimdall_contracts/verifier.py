"""Verifier verdict schema. Per docs/RESEARCH-PROPOSAL.md §4.5 + Appendix B.

The verdict is the single object returned by the two-stage verifier. If
`accepted` is True, then by Theorem 1a (§4.6) the realised profit satisfies
P(pi >= tau) >= 1 - alpha, conditional on exchangeability holding; under
Theorem 1b the long-run miscoverage is bounded by alpha.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

StageFailed = Literal["physical", "conformal"]


class PhysicalViolation(BaseModel):
    """Structured error describing which physical constraint was binding (§4.5 Stage 1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    constraint: Literal[
        "position_envelope",
        "ramp_limit",
        "soc_floor",
        "soc_ceiling",
        "cash_floor",
        "gate_closure",
        "bid_tick_size",
    ]
    current_value: float
    bound_value: float
    suggestion: str


class VerifierVerdict(BaseModel):
    """The verdict the verifier service returns. Authoritative — see Appendix B."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool
    stage_failed: StageFailed | None = None
    physical_violation: PhysicalViolation | None = None
    worst_case_profit_eur: float | None = Field(
        default=None,
        description="pi_min(a) over the conformal interval; required if Stage 2 ran.",
    )
    threshold_eur: float | None = Field(
        default=None,
        description="tau, the minimum-acceptable worst-case profit (e.g. -100 EUR/MWh).",
    )
    alpha: float | None = Field(
        default=None, description="Conformal miscoverage rate used; default 0.1."
    )
    retry_suggestion: str | None = None
