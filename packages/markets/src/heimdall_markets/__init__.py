"""Nordic mFRR / DA / ID market math. Shared by the verifier (worst-case
profit evaluation) and the market simulator (clearing). Centralised so that
Theorem 1's "same profit function" assumption is structurally true (see
docs/RESEARCH-PROPOSAL.md §4.6 footnote on `pi(a, p)`)."""

from heimdall_markets.profit import realized_profit, worst_case_profit

__all__ = ["realized_profit", "worst_case_profit"]
