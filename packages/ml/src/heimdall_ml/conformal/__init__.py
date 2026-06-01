"""Conformal calibrators — Theorem 1a (split-CP), Theorem 1b (online ACI),
and EnbPI (the §5.4 A8 third comparison point).

See docs/RESEARCH-PROPOSAL.md §4.6 for the Theorem 1a/1b formal statements + proofs.
The three calibrators have *different scopes*: split-CP gives a finite-sample,
per-decision guarantee under exchangeability; ACI gives a long-run guarantee
without exchangeability; EnbPI is asymptotic-marginal under bagging. Do not
conflate them.
"""

from heimdall_ml.conformal.aci import AdaptiveConformalInference
from heimdall_ml.conformal.enbpi import EnbPIResult, enbpi_intervals
from heimdall_ml.conformal.split_cp import SplitConformal, fit_quantile, predict_in_band

__all__ = [
    "AdaptiveConformalInference",
    "EnbPIResult",
    "SplitConformal",
    "enbpi_intervals",
    "fit_quantile",
    "predict_in_band",
]
