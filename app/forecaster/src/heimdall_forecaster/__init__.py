"""Forecaster service. Day-1 default: F9 (TimesFM-2.5 zero-shot).

Per docs/RESEARCH-PROPOSAL.md §4.2.2: every forecaster is uncertainty-aware. F9
emits 10 native quantiles; we layer optional split-CP on top for finite-sample
guarantees (Theorem 1a, §4.6).
"""

from heimdall_forecaster.fallback import AR1FallbackForecaster
from heimdall_forecaster.timesfm_wrapper import TimesFMForecaster, available

__all__ = ["AR1FallbackForecaster", "TimesFMForecaster", "available"]
