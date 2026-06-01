"""Verifier FastAPI service. POST /verify orchestrates physical -> conformal,
in that order, per docs/RESEARCH-PROPOSAL.md §4.5.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from heimdall_contracts import BidAction, ConformalInterval, VerifierVerdict
from heimdall_verifier.conformal import conformal_check
from heimdall_verifier.physical import (
    AssetSpec,
    AssetState,
    physical_check,
)


class _AssetSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    q_max_mw: float
    ramp_mw_per_min: float
    storage_mwh: float
    cop: float
    loss_per_quarter: float = 0.005
    bid_tick_eur: float = 0.01

    def to_dataclass(self) -> AssetSpec:
        return AssetSpec(**self.model_dump())


class _AssetStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position_mw: float
    last_delta_mw: float
    soc_mwh: float
    cash_eur: float
    now_utc: datetime
    gate_closure_utc: datetime

    def to_dataclass(self) -> AssetState:
        return AssetState(**self.model_dump())


class VerifyRequest(BaseModel):
    """Input to POST /verify."""

    model_config = ConfigDict(extra="forbid")

    bid: BidAction
    spec: _AssetSpecModel
    state: _AssetStateModel
    interval: ConformalInterval
    tau_eur: float = -100.0


def verify(req: VerifyRequest) -> VerifierVerdict:
    """Pure two-stage verifier (no I/O). Used both by the FastAPI route and
    by integration tests."""
    spec = req.spec.to_dataclass()
    state = req.state.to_dataclass()

    pv = physical_check(req.bid, spec, state)
    if pv is not None:
        return VerifierVerdict(
            accepted=False,
            stage_failed="physical",
            physical_violation=pv,
            alpha=req.interval.alpha,
            threshold_eur=req.tau_eur,
            retry_suggestion=pv.suggestion,
        )

    accepted, pi_min = conformal_check(req.bid, req.interval, req.tau_eur)
    if not accepted:
        return VerifierVerdict(
            accepted=False,
            stage_failed="conformal",
            worst_case_profit_eur=pi_min,
            threshold_eur=req.tau_eur,
            alpha=req.interval.alpha,
            retry_suggestion=(
                f"worst-case profit {pi_min:.2f} EUR < tau {req.tau_eur:.2f}; "
                "reduce volume, raise sell price, or lower buy price"
            ),
        )
    return VerifierVerdict(
        accepted=True,
        worst_case_profit_eur=pi_min,
        threshold_eur=req.tau_eur,
        alpha=req.interval.alpha,
    )


# `asdict` re-export so callers don't need to know dataclasses lives elsewhere.
__all__ = ["VerifyRequest", "asdict", "verify"]


# --- FastAPI surface --------------------------------------------------------

app = FastAPI(title="heimdall-verifier", version="0.0.1")


@app.post("/verify", response_model=VerifierVerdict)
def verify_route(req: VerifyRequest) -> VerifierVerdict:
    return verify(req)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
