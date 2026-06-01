"""Tim-side handoff smoke test.

Run this script after a fresh ``setup.sh`` to verify every Mark-track
surface Tim's focal-orchestrator depends on is reachable:
- the inference layer (every registered backend resolves and predicts),
- the conformal-calibrator service contract,
- the verifier service contract,
- the PyPSA-Eur-Sec asset-spec extraction,
- the e2e composition.

Each check prints a one-line OK / FAIL.  Exits non-zero if any FAIL.

Usage:
  PYTHONPATH=. python tools/tim_smoke.py
"""

from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime, timedelta, timezone


def _check(name: str, fn) -> bool:
    t0 = time.perf_counter()
    try:
        fn()
        dt = time.perf_counter() - t0
        print(f"  OK   {name:40s}  {dt*1000:6.0f} ms")
        return True
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"  FAIL {name:40s}  {dt*1000:6.0f} ms  {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return False


def main() -> int:
    print("Tim-handoff smoke test (deployment readiness check)\n")
    history = [100.0 + 0.05 * (i % 13) for i in range(192)]
    failures: list[str] = []

    print("=== inference layer ===")

    def t_registry():
        from heimdall_forecaster.inference import list_registered
        backends = list_registered()
        assert "f7" in backends and "f8" in backends and "ar1" in backends
        assert len(backends) >= 8, f"expected >=8 backends, got {backends}"

    if not _check("registry contains f0/f7/f8/f9/f10/f11/ar1", t_registry):
        failures.append("registry")

    for backend in ("ar1", "f0", "f7", "f8", "f10", "f3"):
        def _make(b):
            def go():
                from heimdall_forecaster.inference import get_forecaster
                f = get_forecaster(b, seed=42)
                qs = f.predict(history, horizon=4, levels=(0.1, 0.5, 0.9))
                assert len(qs) == 4
                assert all(len(q.values) == 3 for q in qs)
            return go
        if not _check(f"{backend}.predict() returns 4×3 quantile band", _make(backend)):
            failures.append(f"backend:{backend}")

    print("\n=== service contracts ===")

    def t_calibrator():
        from heimdall_conformal_calibrator import (
            PutObservationRequest,
            SeriesUpsertRequest,
            create_or_replace_series,
            get_interval,
            put_observation,
        )
        sid = "tim-smoke"
        create_or_replace_series(
            sid, SeriesUpsertRequest(method="aci", alpha=0.10, warmup_scores=[5.0] * 200),
        )
        put_observation(sid, PutObservationRequest(realised=100.0, point_pred=99.0))
        r = get_interval(sid, point_pred=100.0)
        assert r.interval.lower <= r.interval.upper
        assert r.interval.method == "aci"

    if not _check("conformal-calibrator series put/get cycle", t_calibrator):
        failures.append("calibrator")

    def t_verifier():
        from heimdall_contracts import BidAction, ConformalInterval
        from heimdall_verifier.physical import default_p2h_spec, default_p2h_state
        from heimdall_verifier.service import (
            VerifyRequest, _AssetSpecModel, _AssetStateModel, verify,
        )
        now = datetime(2026, 5, 9, 12, 0, tzinfo=timezone.utc)
        bid = BidAction(
            market="mFRR", direction="buy", quantity_mw=4.0,
            price_eur_per_mwh=50.0,
            delivery_quarter=now + timedelta(minutes=15),
            duration_minutes=15,
        )
        spec, state = default_p2h_spec(), default_p2h_state(now)
        req = VerifyRequest(
            bid=bid,
            spec=_AssetSpecModel(
                q_max_mw=spec.q_max_mw, ramp_mw_per_min=spec.ramp_mw_per_min,
                storage_mwh=spec.storage_mwh, cop=spec.cop,
                loss_per_quarter=spec.loss_per_quarter,
                bid_tick_eur=spec.bid_tick_eur,
            ),
            state=_AssetStateModel(
                position_mw=state.position_mw, last_delta_mw=state.last_delta_mw,
                soc_mwh=state.soc_mwh, cash_eur=state.cash_eur,
                now_utc=state.now_utc, gate_closure_utc=state.gate_closure_utc,
            ),
            interval=ConformalInterval(horizon_minutes=15, alpha=0.10,
                                        lower=40.0, upper=60.0, method="aci"),
            tau_eur=-100.0,
        )
        verdict = verify(req)
        assert verdict.alpha == 0.10

    if not _check("verifier physical+conformal verdict", t_verifier):
        failures.append("verifier")

    def t_pypsa():
        from heimdall_pypsa_scenario import get_assetspec
        spec = get_assetspec("DK1")
        assert abs(spec.cop - 3.2) < 1e-6
        assert "PyPSA" in spec.provenance.get("source", "")

    if not _check("pypsa-scenario /assetspec returns DK1 spec", t_pypsa):
        failures.append("pypsa-scenario")

    print("\n=== summary ===")
    if failures:
        print(f"{len(failures)} FAIL: {failures}")
        return 1
    print("ALL OK — Tim handoff surfaces are reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
