"""Focal-orchestrator stub. Coordinates the forecaster -> verifier path. The
full multi-specialist team (forecaster + risk + regulator + quoter) is sprint
days 10-11 (§7.3); day-1 ships only the deterministic verifier wrapper used
by the orchestrator's E2E smoke test."""

from heimdall_focal_orchestrator.pipeline import end_to_end_decision

__all__ = ["end_to_end_decision"]
