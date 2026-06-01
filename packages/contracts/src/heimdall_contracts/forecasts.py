"""Forecast schemas — every forecaster in the F0..F11 zoo MUST emit one of these.

Per docs/RESEARCH-PROPOSAL.md §4.2.2: every forecaster is uncertainty-aware. The
verifier (Theorem 1a / 1b, §4.6) routes coverage through these intervals; a
point forecast is therefore not a valid output of any focal-path forecaster.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuantileForecast(BaseModel):
    """A multi-quantile forecast for a single horizon.

    `levels` are quantile levels in (0, 1) ascending; `values` are the matching
    predicted values. Length must match. The median (0.5) is recommended but not
    required — TimesFM-2.5 (F9) emits 10 quantiles natively.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon_minutes: int = Field(ge=1)
    levels: tuple[float, ...]
    values: tuple[float, ...]

    @field_validator("levels")
    @classmethod
    def _ascending_in_unit_interval(cls, v: tuple[float, ...]) -> tuple[float, ...]:
        if not v:
            raise ValueError("levels must be non-empty")
        if any(not (0.0 < q < 1.0) for q in v):
            raise ValueError("each quantile level must lie in (0, 1)")
        if list(v) != sorted(v):
            raise ValueError("levels must be ascending")
        return v


class ConformalInterval(BaseModel):
    """Split-CP or ACI prediction interval at level 1 - alpha.

    Required input to the conformal verifier (docs/RESEARCH-PROPOSAL.md §4.5 Stage 2).
    `lower` and `upper` define the closed interval [ell_t, u_t].
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon_minutes: int = Field(ge=1)
    alpha: float = Field(gt=0.0, lt=1.0, description="Miscoverage rate; 0.1 default.")
    lower: float
    upper: float
    method: Literal["split_cp", "aci", "bocpd_aci", "enbpi"]

    @field_validator("upper")
    @classmethod
    def _upper_ge_lower(cls, v: float, info) -> float:  # type: ignore[no-untyped-def]
        lower = info.data.get("lower")
        if lower is not None and v < lower:
            raise ValueError("upper must be >= lower")
        return v


class ActivationForecast(BaseModel):
    """Advisory probabilistic activation forecast.

    This contract is deliberately separate from ``ConformalInterval``. Activation
    forecasts can inform society/simulator decisions, but the verifier theorem
    still routes only through price intervals.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    issued_at: datetime
    zone: Literal["DK1", "DK2"]
    horizon_minutes: int = Field(ge=1)
    p_up: float = Field(ge=0.0, le=1.0)
    p_down: float = Field(ge=0.0, le=1.0)
    p_neutral: float = Field(ge=0.0, le=1.0)
    volume_quantiles_mwh: QuantileForecast
    conformal: ConformalInterval | None = None
    source_model: str
    leakage_guard: Literal["historical_only", "agent_visible_context"] = "historical_only"

    @field_validator("p_neutral")
    @classmethod
    def _probabilities_sum_to_one(cls, v: float, info) -> float:  # type: ignore[no-untyped-def]
        p_up = info.data.get("p_up")
        p_down = info.data.get("p_down")
        if p_up is not None and p_down is not None:
            total = p_up + p_down + v
            if abs(total - 1.0) > 1e-6:
                raise ValueError("activation direction probabilities must sum to 1")
        return v


class _BaseSeriesForecast(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    issued_at: datetime
    series_id: str
    quantiles: tuple[QuantileForecast, ...]
    conformal: ConformalInterval | None = None


class PriceForecast(_BaseSeriesForecast):
    """Forecast of clearing price (EUR/MWh) for one bidding zone."""

    zone: Literal["DK1", "DK2"]


class LoadForecast(_BaseSeriesForecast):
    """Forecast of zonal load (MW)."""

    zone: Literal["DK1", "DK2"]


class WeatherForecast(BaseModel):
    """A multi-variable weather forecast bundle (wind, temperature, irradiance).

    Each variable carries its own quantile bundle. Used by F1..F11 to build
    multivariate forecasts via XReg covariates (TimesFM-2.5 style).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    issued_at: datetime
    location: str
    wind_speed_ms: QuantileForecast | None = None
    temperature_c: QuantileForecast | None = None
    solar_irradiance_w_m2: QuantileForecast | None = None
