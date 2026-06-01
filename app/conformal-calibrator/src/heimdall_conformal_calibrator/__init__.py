"""Heimdall conformal-calibrator service.

Implements docs/RESEARCH-PROPOSAL.md §6.2 service #8.  Receives a stream of
``(point_pred, realised)`` updates per series and serves the current
``ConformalInterval`` on demand.  Holds per-series calibrator state in
process memory; the verifier polls this service every tick rather than
embedding calibrator state itself.

Calibrators are *swappable*: split-CP (Theorem 1a, exchangeable),
online ACI (Theorem 1b, regime-shift-resilient), EnbPI (A8 baseline),
BOCPD-ACI (Theorem 1c, detection-aware).  The service exposes a
``method`` field per series so different focal agents can opt into
different theorem regimes.
"""

from .service import (
    CalibratorMethod,
    PutObservationRequest,
    SeriesIntervalResponse,
    SeriesState,
    SeriesUpsertRequest,
    app,
    create_or_replace_series,
    get_interval,
    put_observation,
)

__all__ = [
    "CalibratorMethod",
    "PutObservationRequest",
    "SeriesIntervalResponse",
    "SeriesState",
    "SeriesUpsertRequest",
    "app",
    "create_or_replace_series",
    "get_interval",
    "put_observation",
]
