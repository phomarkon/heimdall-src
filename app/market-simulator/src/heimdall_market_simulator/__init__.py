"""Minimal mFRR mock simulator. Day-1 stub.

# DEVIATION: the full PyPSA-Eur-Sec integration is sprint days 3-4 (§7.3).
# Day 1 ships a deterministic, in-memory step function that emits market
# state, accepts settled bids, and returns realised activations. Sufficient
# to drive end-to-end smoke tests against `verifier` and `agent-runner`.
"""

from heimdall_market_simulator.mock_market import MockMFRRMarket

__all__ = ["MockMFRRMarket"]
