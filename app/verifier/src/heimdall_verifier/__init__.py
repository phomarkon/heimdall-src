"""Two-stage verifier service. Sits between the focal agent's quoter and the
market submission endpoint (docs/RESEARCH-PROPOSAL.md §4.5)."""

from heimdall_verifier.conformal import conformal_check
from heimdall_verifier.physical import (
    AssetSpec,
    AssetState,
    physical_check,
)
from heimdall_verifier.service import VerifyRequest, verify

__all__ = [
    "AssetSpec",
    "AssetState",
    "VerifyRequest",
    "conformal_check",
    "physical_check",
    "verify",
]
