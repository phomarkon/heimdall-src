"""End-to-end pipeline test: forecaster → calibrator → verifier.

Exercises the full Mark-track service surface in one go, the way Tim's
focal-orchestrator will compose them at runtime:

  1. Pull a synthetic price history.
  2. Get a quantile forecast from the inference service (`f7` or `ar1`).
  3. Push the realised price + q50 to the calibrator service to get a
     conformal interval.
  4. Submit a candidate ``BidAction`` plus the spec, state, and interval
     to the verifier service.  Assert the verdict is well-formed.

Failure of this test means the pipeline's contracts are misaligned.
This is the test Tim would write first if Mark's surface were a
black-box dependency.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from packages.pypsa_adapter.eursec_costs import DEFAULT_COSTS_CSV

from heimdall_conformal_calibrator import (
    PutObservationRequest,
    SeriesUpsertRequest,
    create_or_replace_series,
    get_interval,
    put_observation,
)
from heimdall_contracts import BidAction, ConformalInterval
from heimdall_forecaster.inference import get_forecaster
from heimdall_verifier.physical import default_p2h_spec, default_p2h_state
from heimdall_verifier.scenario_loader import (
    assetspec_from_pypsa_eursec_dk_network,
)
from heimdall_verifier.service import (
    VerifyRequest,
    _AssetSpecModel,
    _AssetStateModel,
    verify,
)

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("backend", ["ar1", "f0"])
def test_full_pipeline_well_formed(backend: str) -> None:
    rng = np.random.default_rng(42)
    history = list(100.0 + rng.normal(0, 5.0, 192))

    # 1. Forecast.
    f = get_forecaster(backend)
    quantiles = f.predict(history, horizon=4, levels=(0.1, 0.5, 0.9))
    q0 = quantiles[0]
    q10, q50, q90 = q0.values

    # 2. Calibrator: warm up with synthetic residuals.
    sid = f"e2e-{backend}"
    create_or_replace_series(
        sid,
        SeriesUpsertRequest(
            method="aci",
            alpha=0.10,
            warmup_scores=list(np.abs(rng.normal(0, 5, 200))),
        ),
    )
    put_observation(sid, PutObservationRequest(realised=99.0, point_pred=q50))
    cal_resp = get_interval(sid, point_pred=q50)
    assert cal_resp.interval.lower <= cal_resp.interval.upper

    # 3. Verifier.
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    bid = BidAction(
        market="mFRR",
        direction="sell",
        quantity_mw=4.0,
        price_eur_per_mwh=cal_resp.interval.lower - 5.0,  # aggressive: forces fill
        delivery_quarter=now + timedelta(minutes=15),
        duration_minutes=15,
    )
    spec_d = default_p2h_spec()
    state_d = default_p2h_state(now)
    req = VerifyRequest(
        bid=bid,
        spec=_AssetSpecModel(
            q_max_mw=spec_d.q_max_mw,
            ramp_mw_per_min=spec_d.ramp_mw_per_min,
            storage_mwh=spec_d.storage_mwh,
            cop=spec_d.cop,
            loss_per_quarter=spec_d.loss_per_quarter,
            bid_tick_eur=spec_d.bid_tick_eur,
        ),
        state=_AssetStateModel(
            position_mw=state_d.position_mw,
            last_delta_mw=state_d.last_delta_mw,
            soc_mwh=state_d.soc_mwh,
            cash_eur=state_d.cash_eur,
            now_utc=state_d.now_utc,
            gate_closure_utc=state_d.gate_closure_utc,
        ),
        interval=cal_resp.interval,
        tau_eur=-100.0,
    )
    verdict = verify(req)
    assert verdict.alpha == 0.10
    assert verdict.threshold_eur == -100.0
    # Either accepted or rejected with a structured reason — both shapes valid.
    if not verdict.accepted:
        assert verdict.stage_failed in {"physical", "conformal"}


@pytest.mark.skipif(
    not Path(DEFAULT_COSTS_CSV).exists(),
    reason="costs_2030.csv not pulled; run setup.sh or curl into data/raw/pypsa_eursec/",
)
def test_pypsa_eursec_assetspec_composes_into_verifier() -> None:
    """The PyPSA-Eur-Sec-derived spec must be a valid input to /verify."""
    spec = assetspec_from_pypsa_eursec_dk_network(zone="DK1")
    now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
    bid = BidAction(
        market="mFRR",
        direction="buy",
        quantity_mw=4.0,
        price_eur_per_mwh=50.0,
        delivery_quarter=now + timedelta(minutes=15),
        duration_minutes=15,
    )
    interval = ConformalInterval(
        horizon_minutes=15, alpha=0.10, lower=40.0, upper=60.0, method="aci",
    )
    state_d = default_p2h_state(now)
    req = VerifyRequest(
        bid=bid,
        spec=_AssetSpecModel(
            q_max_mw=spec.q_max_mw,
            ramp_mw_per_min=spec.ramp_mw_per_min,
            storage_mwh=spec.storage_mwh,
            cop=spec.cop,
            loss_per_quarter=spec.loss_per_quarter,
            bid_tick_eur=spec.bid_tick_eur,
        ),
        state=_AssetStateModel(
            position_mw=state_d.position_mw,
            last_delta_mw=state_d.last_delta_mw,
            soc_mwh=state_d.soc_mwh,
            cash_eur=state_d.cash_eur,
            now_utc=state_d.now_utc,
            gate_closure_utc=state_d.gate_closure_utc,
        ),
        interval=interval,
        tau_eur=-100.0,
    )
    verdict = verify(req)
    assert verdict.alpha == 0.10
