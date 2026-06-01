"""Forecaster FastAPI service. POST /forecast.

Depends only on the modular ``inference/`` layer — never on a concrete
backend.  Adding a new zoo member requires zero edits here; just drop a
new file under ``inference/backends/`` and decorate its loader with
``@register("name")``.

Backend selection priority:
  1. ``backend`` field on the incoming ``ForecastRequest`` (Tim override)
  2. ``HEIMDALL_FORECASTER_BACKEND`` env var (deployment default)
  3. ``f7`` if its checkpoint can be hydrated, else ``ar1`` (graceful)
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from heimdall_contracts import QuantileForecast
from heimdall_forecaster.inference import (
    Forecaster,
    describe,
    get_forecaster,
    list_registered,
)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    history: list[float] = Field(min_length=8)
    horizon: int = Field(ge=1, le=1024)
    levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    backend: str = "auto"
    seed: int = 42


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend_used: str
    quantiles: list[QuantileForecast]


def _resolve_backend(req: ForecastRequest) -> str:
    if req.backend != "auto":
        return req.backend.lower()
    env = os.environ.get("HEIMDALL_FORECASTER_BACKEND", "").lower().strip()
    if env:
        return env
    return "f7" if "f7" in list_registered() else "ar1"


def forecast(req: ForecastRequest) -> ForecastResponse:
    name = _resolve_backend(req)
    if name not in list_registered():
        raise HTTPException(
            status_code=400,
            detail=f"unknown backend {name!r}; known: {list_registered()}",
        )
    try:
        f: Forecaster = get_forecaster(name, req.seed)
    except FileNotFoundError as e:
        # Hydrate failure — fall back to ar1 transparently.
        f = get_forecaster("ar1", req.seed)
        name = "ar1"
    qfs = f.predict(req.history, horizon=req.horizon, levels=tuple(req.levels))
    return ForecastResponse(backend_used=name, quantiles=qfs)


app = FastAPI(title="heimdall-forecaster", version="0.1.0")


@app.post("/forecast", response_model=ForecastResponse)
def forecast_route(req: ForecastRequest) -> ForecastResponse:
    return forecast(req)


@app.get("/healthz")
def healthz() -> dict:
    backend_default = os.environ.get("HEIMDALL_FORECASTER_BACKEND", "auto")
    return {
        "status": "ok",
        "default_backend": backend_default,
        "registered_backends": [
            {"name": n, "description": describe(n)} for n in list_registered()
        ],
    }
