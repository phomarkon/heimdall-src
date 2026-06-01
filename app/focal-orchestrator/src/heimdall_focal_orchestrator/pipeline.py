"""End-to-end smoke pipeline: history -> forecast -> conformal interval ->
verifier verdict. Used by `notebooks/` exploration and by the day-1 E2E test.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from numpy.typing import ArrayLike

from heimdall_contracts import BidAction, ConformalInterval, VerifierVerdict
from heimdall_forecaster import AR1FallbackForecaster
from heimdall_ml.conformal import AdaptiveConformalInference
from heimdall_verifier.physical import default_p2h_spec, default_p2h_state
from heimdall_verifier.service import VerifyRequest, verify


def end_to_end_decision(
    history: ArrayLike,
    candidate_bid: BidAction,
    *,
    alpha: float = 0.1,
    tau_eur: float = -100.0,
    seed: int = 13,
) -> VerifierVerdict:
    """Forecast next quarter, calibrate online ACI, run two-stage verifier."""
    h = np.asarray(history, dtype=np.float64).ravel()
    fc = AR1FallbackForecaster(seed=seed, levels=(alpha / 2, 0.5, 1.0 - alpha / 2))
    next_q = fc.fit_predict_quantiles(h, horizon=1)[0]
    lo, _, hi = next_q.values

    # Warm-start ACI with absolute residuals from the AR(1) fit (cheap proxy).
    aci = AdaptiveConformalInference(alpha=alpha, gamma=0.05)
    residuals = np.abs(np.diff(h))
    aci.warm_start(residuals)

    interval = ConformalInterval(
        horizon_minutes=15,
        alpha=alpha,
        lower=float(lo),
        upper=float(hi),
        method="aci",
    )

    spec = default_p2h_spec()
    state = default_p2h_state(now=datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc))
    req = VerifyRequest(
        bid=candidate_bid,
        spec={  # type: ignore[arg-type]
            "q_max_mw": spec.q_max_mw,
            "ramp_mw_per_min": spec.ramp_mw_per_min,
            "storage_mwh": spec.storage_mwh,
            "cop": spec.cop,
            "loss_per_quarter": spec.loss_per_quarter,
            "bid_tick_eur": spec.bid_tick_eur,
        },
        state={  # type: ignore[arg-type]
            "position_mw": state.position_mw,
            "last_delta_mw": state.last_delta_mw,
            "soc_mwh": state.soc_mwh,
            "cash_eur": state.cash_eur,
            "now_utc": state.now_utc,
            "gate_closure_utc": state.gate_closure_utc,
        },
        interval=interval,
        tau_eur=tau_eur,
    )
    return verify(req)
