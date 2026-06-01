"""Conformal-calibrator FastAPI service.

API:
  PUT  /series/{id}                    create or replace a series, choose method
  POST /series/{id}/observation        push a (realised, point_pred) update
  GET  /series/{id}/interval           current 1-α conformal interval at point_pred
  GET  /series/{id}/state              full state dump (debug / restore)
  GET  /healthz                        liveness

Methods (docs/RESEARCH-PROPOSAL.md §4.6, §5.4):
  - ``split_cp`` : frozen calibration buffer (Theorem 1a)
  - ``aci``      : online Gibbs–Candès (Theorem 1b)  ← default
  - ``bocpd_aci``: ACI with detection-driven buffer reset (Theorem 1c)

State is process-local.  In multi-replica deployments use sticky
sessions or push to TimescaleDB via ``replay-store``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from heimdall_contracts import ConformalInterval
from heimdall_ml.conformal.bocpd import BOCPD
from pydantic import BaseModel, ConfigDict, Field

CalibratorMethod = Literal["split_cp", "aci", "bocpd_aci"]
DEFAULT_METHOD: CalibratorMethod = "aci"

ALPHA_FLOOR = 1e-3
ALPHA_CEIL = 1 - 1e-3
BOCPD_BUFFER_TICKS = 192  # 2 days at 15-min resolution
FALLBACK_INTERVAL_WIDTH = 100.0


class SeriesUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: CalibratorMethod = DEFAULT_METHOD
    alpha: float = Field(0.10, gt=0.0, lt=1.0)
    gamma: float = Field(0.05, gt=0.0, le=1.0, description="ACI learning rate")
    horizon_minutes: int = Field(15, ge=1)
    warmup_scores: list[float] | None = None
    bocpd_mean_run_length: float = 200.0


class PutObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    realised: float
    point_pred: float


class SeriesIntervalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    interval: ConformalInterval
    method: CalibratorMethod
    n_observations: int
    alpha_t: float
    last_reset_t: int | None = None


class SeriesState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: CalibratorMethod
    alpha_target: float
    alpha_t: float
    gamma: float
    horizon_minutes: int
    n_observations: int
    last_reset_t: int | None
    score_buffer_size: int


class _CalibrationStrategy:
    """Base class for conformal calibration strategies (Theorems 1a/1b/1c)."""

    def update(self, score: float, residual: float, entry: _SeriesEntry) -> None:
        raise NotImplementedError

    def effective_alpha(self, entry: _SeriesEntry) -> float:
        raise NotImplementedError


class _SplitCPStrategy(_CalibrationStrategy):
    """Frozen calibration buffer — Theorem 1a (finite-sample coverage)."""

    def update(self, score: float, residual: float, entry: _SeriesEntry) -> None:
        entry.t += 1

    def effective_alpha(self, entry: _SeriesEntry) -> float:
        return entry.alpha_target


class _ACIStrategy(_CalibrationStrategy):
    """Online Gibbs-Candes ACI — Theorem 1b (long-run coverage)."""

    def update(self, score: float, residual: float, entry: _SeriesEntry) -> None:
        if entry.scores:
            a = float(np.clip(entry.alpha_t, ALPHA_FLOOR, ALPHA_CEIL))
            q = float(np.quantile(entry.scores, 1.0 - a))
            err = 0.0 if score <= q else 1.0
        else:
            err = 0.0
        entry.scores.append(score)
        entry.alpha_t = float(
            np.clip(entry.alpha_t + entry.gamma * (entry.alpha_target - err), ALPHA_FLOOR, ALPHA_CEIL)
        )
        entry.t += 1

    def effective_alpha(self, entry: _SeriesEntry) -> float:
        return float(np.clip(entry.alpha_t, ALPHA_FLOOR, ALPHA_CEIL))


class _BOCPDACIStrategy(_ACIStrategy):
    """ACI with detection-driven buffer reset — Theorem 1c."""

    def __init__(self, mean_run_length: float = 200.0) -> None:
        self._bocpd = BOCPD(mean_run_length=mean_run_length)

    def update(self, score: float, residual: float, entry: _SeriesEntry) -> None:
        super().update(score, residual, entry)
        r = self._bocpd.step(residual)
        if r.detected_change and (entry.last_reset_t is None or entry.t - entry.last_reset_t > BOCPD_BUFFER_TICKS):
            entry.scores = list(entry.scores[-BOCPD_BUFFER_TICKS:])
            entry.alpha_t = entry.alpha_target
            entry.last_reset_t = entry.t


_STRATEGIES: dict[CalibratorMethod, type[_CalibrationStrategy]] = {
    "split_cp": _SplitCPStrategy,
    "aci": _ACIStrategy,
}


def _make_strategy(req: SeriesUpsertRequest) -> _CalibrationStrategy:
    if req.method == "bocpd_aci":
        return _BOCPDACIStrategy(mean_run_length=req.bocpd_mean_run_length)
    return _STRATEGIES[req.method]()


class _SeriesEntry:
    """In-memory state for one calibrated series."""

    def __init__(self, req: SeriesUpsertRequest) -> None:
        self.method: CalibratorMethod = req.method
        self.alpha_target: float = req.alpha
        self.gamma: float = req.gamma
        self.horizon_minutes: int = req.horizon_minutes
        self.alpha_t: float = req.alpha
        self.scores: list[float] = list(req.warmup_scores or [])
        self.t: int = len(self.scores)
        self.last_reset_t: int | None = None
        self._strategy: _CalibrationStrategy = _make_strategy(req)

    def update(self, realised: float, point_pred: float) -> None:
        score = abs(realised - point_pred)
        residual = realised - point_pred
        self._strategy.update(score, residual, self)

    def interval(self, point_pred: float) -> ConformalInterval:
        if not self.scores:
            return ConformalInterval(
                horizon_minutes=self.horizon_minutes,
                alpha=self.alpha_target,
                lower=point_pred - FALLBACK_INTERVAL_WIDTH,
                upper=point_pred + FALLBACK_INTERVAL_WIDTH,
                method=self.method,
            )
        a_eff = self._strategy.effective_alpha(self)
        q = float(np.quantile(self.scores, 1.0 - a_eff))
        return ConformalInterval(
            horizon_minutes=self.horizon_minutes,
            alpha=self.alpha_target,
            lower=point_pred - q,
            upper=point_pred + q,
            method=self.method,
        )


_SERIES: dict[str, _SeriesEntry] = {}


# ---- pure functions exposed for tests -------------------------------------


def create_or_replace_series(series_id: str, req: SeriesUpsertRequest) -> SeriesState:
    _SERIES[series_id] = _SeriesEntry(req)
    e = _SERIES[series_id]
    return SeriesState(
        method=e.method,
        alpha_target=e.alpha_target,
        alpha_t=e.alpha_t,
        gamma=e.gamma,
        horizon_minutes=e.horizon_minutes,
        n_observations=e.t,
        last_reset_t=e.last_reset_t,
        score_buffer_size=len(e.scores),
    )


def put_observation(series_id: str, obs: PutObservationRequest) -> SeriesState:
    if series_id not in _SERIES:
        raise KeyError(series_id)
    e = _SERIES[series_id]
    e.update(obs.realised, obs.point_pred)
    return SeriesState(
        method=e.method,
        alpha_target=e.alpha_target,
        alpha_t=e.alpha_t,
        gamma=e.gamma,
        horizon_minutes=e.horizon_minutes,
        n_observations=e.t,
        last_reset_t=e.last_reset_t,
        score_buffer_size=len(e.scores),
    )


def get_interval(series_id: str, point_pred: float) -> SeriesIntervalResponse:
    if series_id not in _SERIES:
        raise KeyError(series_id)
    e = _SERIES[series_id]
    return SeriesIntervalResponse(
        interval=e.interval(point_pred),
        method=e.method,
        n_observations=e.t,
        alpha_t=e.alpha_t,
        last_reset_t=e.last_reset_t,
    )


# ---- FastAPI surface ------------------------------------------------------


app = FastAPI(title="heimdall-conformal-calibrator", version="0.1.0")


@app.put("/series/{series_id}", response_model=SeriesState)
def put_series(series_id: str, req: SeriesUpsertRequest) -> SeriesState:
    return create_or_replace_series(series_id, req)


@app.post("/series/{series_id}/observation", response_model=SeriesState)
def post_obs(series_id: str, obs: PutObservationRequest) -> SeriesState:
    try:
        return put_observation(series_id, obs)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"series {series_id!r} not found")


@app.get("/series/{series_id}/interval", response_model=SeriesIntervalResponse)
def get_int(series_id: str, point_pred: float) -> SeriesIntervalResponse:
    try:
        return get_interval(series_id, point_pred)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"series {series_id!r} not found")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "n_series": len(_SERIES)}
