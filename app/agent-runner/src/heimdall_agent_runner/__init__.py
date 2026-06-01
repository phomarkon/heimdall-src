"""Agent runner. Day-1 stub.

# DEVIATION: per task brief, the day-1 baseline is a *reflex* policy that
# ignores the LLM call entirely. The vLLM integration (Qwen3.6-35B-A3B FP8 on
# B200, docs/RESEARCH-PROPOSAL.md §4.2.2) is deferred to days 5-6 of the sprint
# (§7.3). The reflex policy is also a useful B6/B7 ablation anchor in §5.2.
"""

from heimdall_agent_runner.reflex import (
    AgentDecisionRequest,
    AgentDecisionResponse,
    decide,
)

__all__ = ["AgentDecisionRequest", "AgentDecisionResponse", "decide"]
