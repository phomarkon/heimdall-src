from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
from typing import Any, Literal

import pandas as pd

from .models import Bid


Direction = Literal["up", "down", "neutral"]
FallbackLevel = Literal[
    "zone_direction_hour",
    "zone_direction",
    "direction_hour",
    "direction",
    "global",
]


@dataclass(frozen=True)
class PriceModelQuality:
    zone: str
    direction: str
    fallback_level: FallbackLevel
    hour_utc: int | None
    sample_count: int
    mae_eur_mwh: float
    r2: float


@dataclass(frozen=True)
class PricePrediction:
    predicted_imbalance_price_eur_mwh: float
    lower_90_eur_mwh: float
    upper_90_eur_mwh: float
    uncertainty_90_eur_mwh: float
    predicted_spread_eur_mwh: float
    slope_eur_per_mwh: float
    intercept_eur_mwh: float
    quality: PriceModelQuality


@dataclass(frozen=True)
class ClearingBidDecision:
    agent_id: str
    asset_id: str
    zone: str
    side: str
    quantity_mwh: float
    limit_price_eur_mwh: float
    accepted: bool
    reason_code: str | None
    settlement_eur: float


@dataclass(frozen=True)
class MFRRClearingResult:
    utc_timestamp: datetime
    zone: str
    direction: Direction
    spot_price_eur_mwh: float
    historical_imbalance_price_eur_mwh: float
    baseline_activation_volume_mwh: float
    accepted_volume_mwh: float
    counterfactual_activation_volume_mwh: float
    counterfactual_imbalance_price_eur_mwh: float
    price_delta_eur_mwh: float
    model_quality: PriceModelQuality
    accepted_bids: list[ClearingBidDecision]
    rejected_bids: list[ClearingBidDecision]
    result_hash: str


@dataclass(frozen=True)
class BacktestReport:
    total_rows: int
    mae_eur_mwh: float
    max_abs_error_eur_mwh: float
    interval_coverage_90: float
    zone_metrics: dict[str, float]
    direction_metrics: dict[str, float]
    result_hash: str


@dataclass(frozen=True)
class _CurveFit:
    zone: str
    direction: str
    fallback_level: FallbackLevel
    hour_utc: int | None
    slope: float
    intercept: float
    sample_count: int
    mae: float
    r2: float
    residual_q90: float


class CalibratedMFRRPriceModel:
    def __init__(self, fits: dict[tuple[str, ...], _CurveFit]) -> None:
        self._fits = fits

    @classmethod
    def fit(cls, rows: pd.DataFrame, *, min_samples: int = 8) -> "CalibratedMFRRPriceModel":
        prepared = _prepare_rows(rows)
        fits: dict[tuple[str, ...], _CurveFit] = {}
        groups: list[tuple[str, str, int | None, FallbackLevel, pd.DataFrame]] = []

        for (zone, direction, hour), group in prepared.groupby(
            ["zone", "direction", "hour_utc"]
        ):
            groups.append(
                (str(zone), str(direction), int(hour), "zone_direction_hour", group)
            )

        for (zone, direction), group in prepared.groupby(["zone", "direction"]):
            groups.append((str(zone), str(direction), None, "zone_direction", group))
        for (direction, hour), group in prepared.groupby(["direction", "hour_utc"]):
            groups.append(("*", str(direction), int(hour), "direction_hour", group))
        for direction, group in prepared.groupby("direction"):
            groups.append(("*", str(direction), None, "direction", group))
        groups.append(("*", "*", None, "global", prepared))

        for zone, direction, hour, fallback_level, group in groups:
            if len(group) < min_samples:
                continue
            fits[(zone, direction)] = _fit_curve(
                group,
                zone=zone,
                direction=direction,
                hour_utc=hour,
                fallback_level=fallback_level,
            )
            if hour is not None:
                fits[(zone, direction, str(hour))] = fits[(zone, direction)]

        if not fits:
            raise ValueError("Not enough mFRR history to fit any price-response curve")
        return cls(fits)

    def predict(
        self,
        *,
        zone: str,
        direction: Direction,
        spot_price_eur_mwh: float,
        activation_volume_mwh: float,
        hour_utc: int | None = None,
    ) -> PricePrediction:
        if direction == "neutral":
            quality = PriceModelQuality(
                zone=zone,
                direction=direction,
                fallback_level="global",
                hour_utc=hour_utc,
                sample_count=0,
                mae_eur_mwh=0.0,
                r2=1.0,
            )
            return PricePrediction(
                predicted_imbalance_price_eur_mwh=round(spot_price_eur_mwh, 6),
                lower_90_eur_mwh=round(spot_price_eur_mwh, 6),
                upper_90_eur_mwh=round(spot_price_eur_mwh, 6),
                uncertainty_90_eur_mwh=0.0,
                predicted_spread_eur_mwh=0.0,
                slope_eur_per_mwh=0.0,
                intercept_eur_mwh=0.0,
                quality=quality,
            )

        fit = self._resolve_fit(zone, direction, hour_utc)
        spread = max(0.0, fit.intercept + fit.slope * activation_volume_mwh)
        price = (
            spot_price_eur_mwh + spread
            if direction == "up"
            else spot_price_eur_mwh - spread
        )
        uncertainty = self._uncertainty_for_fit(fit, direction)
        return PricePrediction(
            predicted_imbalance_price_eur_mwh=round(price, 6),
            lower_90_eur_mwh=round(price - uncertainty, 6),
            upper_90_eur_mwh=round(price + uncertainty, 6),
            uncertainty_90_eur_mwh=round(uncertainty, 6),
            predicted_spread_eur_mwh=round(spread, 6),
            slope_eur_per_mwh=round(fit.slope, 6),
            intercept_eur_mwh=round(fit.intercept, 6),
            quality=PriceModelQuality(
                zone=zone,
                direction=direction,
                fallback_level=fit.fallback_level,
                hour_utc=hour_utc if fit.hour_utc is not None else None,
                sample_count=fit.sample_count,
                mae_eur_mwh=round(fit.mae, 6),
                r2=round(fit.r2, 6),
            ),
        )

    def _resolve_fit(self, zone: str, direction: str, hour_utc: int | None) -> _CurveFit:
        hour_key = str(hour_utc) if hour_utc is not None else None
        keys = []
        if hour_key is not None:
            keys.extend([(zone, direction, hour_key), ("*", direction, hour_key)])
        keys.extend([(zone, direction), ("*", direction), ("*", "*")])
        for key in keys:
            if key in self._fits:
                return self._fits[key]
        raise ValueError(f"No mFRR price-response curve available for {zone}/{direction}")

    def _uncertainty_for_fit(self, fit: _CurveFit, direction: str) -> float:
        candidates = [fit.residual_q90]
        direction_fit = self._fits.get(("*", direction))
        if direction_fit is not None:
            candidates.append(direction_fit.residual_q90)
        global_fit = self._fits.get(("*", "*"))
        if global_fit is not None:
            candidates.append(global_fit.residual_q90)
        return max(candidates)


class MFRRClearingEngine:
    def __init__(self, price_model: CalibratedMFRRPriceModel) -> None:
        self._price_model = price_model

    def clear(self, market_row: dict[str, Any], bids: list[Bid]) -> MFRRClearingResult:
        timestamp = _parse_utc(market_row["utc_timestamp"])
        zone = str(market_row["zone"])
        direction = _activation_direction(float(market_row["satisfied_demand_mw"]))
        spot = float(market_row["spot_price_eur_mwh"])
        historical_price = float(market_row["imbalance_price_eur_mwh"])
        baseline_volume = abs(float(market_row["satisfied_demand_mw"])) * 0.25

        baseline_prediction = self._price_model.predict(
            zone=zone,
            direction=direction,
            spot_price_eur_mwh=spot,
            activation_volume_mwh=baseline_volume,
            hour_utc=timestamp.hour,
        )
        marginal_price = historical_price if direction != "neutral" else spot

        accepted: list[ClearingBidDecision] = []
        rejected: list[ClearingBidDecision] = []
        remaining_volume = baseline_volume
        for bid in _merit_order(bids):
            rejection = _bid_rejection_reason(
                bid,
                timestamp=timestamp,
                zone=zone,
                direction=direction,
                marginal_price=marginal_price,
            )
            if rejection is not None:
                rejected.append(_decision(bid, False, rejection, 0.0))
                continue

            cleared_quantity = min(bid.quantity_mwh, remaining_volume)
            if cleared_quantity <= 1e-9:
                rejected.append(_decision(bid, False, "activation_volume_filled", 0.0))
                continue

            remaining_volume = round(remaining_volume - cleared_quantity, 12)
            accepted.append(
                _decision(
                    bid,
                    True,
                    None,
                    _settlement(
                        side=bid.side,
                        quantity_mwh=cleared_quantity,
                        spot_price_eur_mwh=spot,
                        clearing_price_eur_mwh=marginal_price,
                    ),
                    quantity_mwh=cleared_quantity,
                )
            )

        accepted_volume = round(sum(row.quantity_mwh for row in accepted), 6)
        counterfactual_volume = max(0.0, baseline_volume - accepted_volume)
        counterfactual_price = _anchored_counterfactual_price(
            direction=direction,
            historical_price=historical_price,
            spot_price=spot,
            accepted_volume_mwh=accepted_volume,
            slope_eur_per_mwh=baseline_prediction.slope_eur_per_mwh,
        )
        result = MFRRClearingResult(
            utc_timestamp=timestamp,
            zone=zone,
            direction=direction,
            spot_price_eur_mwh=round(spot, 6),
            historical_imbalance_price_eur_mwh=round(historical_price, 6),
            baseline_activation_volume_mwh=round(baseline_volume, 6),
            accepted_volume_mwh=accepted_volume,
            counterfactual_activation_volume_mwh=round(counterfactual_volume, 6),
            counterfactual_imbalance_price_eur_mwh=round(counterfactual_price, 6),
            price_delta_eur_mwh=round(counterfactual_price - historical_price, 6),
            model_quality=baseline_prediction.quality,
            accepted_bids=accepted,
            rejected_bids=rejected,
            result_hash="",
        )
        return replace(result, result_hash=_hash_payload(asdict(result)))


def backtest_price_model(
    rows: pd.DataFrame, *, train_fraction: float = 0.7, min_samples: int = 8
) -> BacktestReport:
    prepared = _prepare_rows(rows).sort_values(["utc_timestamp", "zone", "direction"])
    split_index = max(1, min(len(prepared) - 1, int(len(prepared) * train_fraction)))
    train = prepared.iloc[:split_index].copy()
    test = prepared.iloc[split_index:].copy()
    model = CalibratedMFRRPriceModel.fit(train, min_samples=min_samples)

    errors: list[dict[str, Any]] = []
    for row in test.to_dict(orient="records"):
        prediction = model.predict(
            zone=str(row["zone"]),
            direction=row["direction"],
            spot_price_eur_mwh=float(row["spot_price_eur_mwh"]),
            activation_volume_mwh=float(row["activation_volume_mwh"]),
            hour_utc=int(row["hour_utc"]),
        )
        error = prediction.predicted_imbalance_price_eur_mwh - float(
            row["target_imbalance_price_eur_mwh"]
        )
        covered = (
            prediction.lower_90_eur_mwh
            <= float(row["target_imbalance_price_eur_mwh"])
            <= prediction.upper_90_eur_mwh
        )
        errors.append(
            {
                "zone": str(row["zone"]),
                "direction": str(row["direction"]),
                "abs_error": abs(error),
                "covered": covered,
            }
        )

    error_frame = pd.DataFrame(errors)
    mae = float(error_frame["abs_error"].mean()) if not error_frame.empty else 0.0
    max_abs_error = float(error_frame["abs_error"].max()) if not error_frame.empty else 0.0
    coverage = float(error_frame["covered"].mean()) if not error_frame.empty else 1.0
    zone_metrics = {
        str(zone): round(float(group["abs_error"].mean()), 6)
        for zone, group in error_frame.groupby("zone")
    }
    direction_metrics = {
        str(direction): round(float(group["abs_error"].mean()), 6)
        for direction, group in error_frame.groupby("direction")
    }
    report = BacktestReport(
        total_rows=len(error_frame),
        mae_eur_mwh=round(mae, 6),
        max_abs_error_eur_mwh=round(max_abs_error, 6),
        interval_coverage_90=round(coverage, 6),
        zone_metrics=zone_metrics,
        direction_metrics=direction_metrics,
        result_hash="",
    )
    return replace(report, result_hash=_hash_payload(asdict(report)))


def _prepare_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "utc_timestamp",
        "zone",
        "satisfied_demand_mw",
        "imbalance_price_eur_mwh",
        "spot_price_eur_mwh",
        "mfrr_marginal_price_up_eur_mwh",
        "mfrr_marginal_price_down_eur_mwh",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"mFRR calibration rows missing required columns: {missing}")

    prepared = rows.copy()
    prepared["utc_timestamp"] = pd.to_datetime(
        prepared["utc_timestamp"], utc=True, errors="coerce"
    )
    numeric_columns = [
        "satisfied_demand_mw",
        "imbalance_price_eur_mwh",
        "spot_price_eur_mwh",
        "mfrr_marginal_price_up_eur_mwh",
        "mfrr_marginal_price_down_eur_mwh",
    ]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=["utc_timestamp", "zone", *numeric_columns])
    prepared["direction"] = prepared["satisfied_demand_mw"].map(_activation_direction)
    prepared = prepared[prepared["direction"] != "neutral"].copy()
    prepared["activation_volume_mwh"] = prepared["satisfied_demand_mw"].abs() * 0.25
    prepared["hour_utc"] = prepared["utc_timestamp"].dt.hour
    prepared["observed_spread_eur_mwh"] = prepared.apply(_observed_spread, axis=1)
    prepared["target_imbalance_price_eur_mwh"] = prepared.apply(
        _target_imbalance_price, axis=1
    )
    return prepared[prepared["activation_volume_mwh"] > 0].reset_index(drop=True)


def _fit_curve(
    group: pd.DataFrame,
    *,
    zone: str,
    direction: str,
    hour_utc: int | None,
    fallback_level: FallbackLevel,
) -> _CurveFit:
    x = group["activation_volume_mwh"].astype(float)
    y = group["observed_spread_eur_mwh"].astype(float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float((x_centered**2).sum())
    slope = (
        0.0
        if denominator == 0.0
        else float((x_centered * y_centered).sum() / denominator)
    )
    intercept = float(y.mean() - slope * x.mean())
    predicted = intercept + slope * x
    residuals = y - predicted
    mae = float(residuals.abs().mean())
    residual_q90 = float(residuals.abs().quantile(0.9))
    total_variance = float((y_centered**2).sum())
    r2 = 1.0 if total_variance == 0.0 else 1.0 - float((residuals**2).sum()) / total_variance
    return _CurveFit(
        zone=zone,
        direction=direction,
        fallback_level=fallback_level,
        hour_utc=hour_utc,
        slope=max(0.0, slope),
        intercept=max(0.0, intercept),
        sample_count=len(group),
        mae=mae,
        r2=r2,
        residual_q90=residual_q90,
    )


def _activation_direction(satisfied_demand_mw: float) -> Direction:
    if satisfied_demand_mw > 0:
        return "up"
    if satisfied_demand_mw < 0:
        return "down"
    return "neutral"


def _observed_spread(row: pd.Series) -> float:
    if row["direction"] == "up":
        return max(
            0.0,
            float(row["mfrr_marginal_price_up_eur_mwh"])
            - float(row["spot_price_eur_mwh"]),
        )
    return max(
        0.0,
        float(row["spot_price_eur_mwh"])
        - float(row["mfrr_marginal_price_down_eur_mwh"]),
    )


def _target_imbalance_price(row: pd.Series) -> float:
    if row["direction"] == "up":
        return float(row["mfrr_marginal_price_up_eur_mwh"])
    return float(row["mfrr_marginal_price_down_eur_mwh"])


def _parse_utc(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("mFRR market timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _merit_order(bids: list[Bid]) -> list[Bid]:
    return sorted(
        bids,
        key=lambda bid: (
            bid.utc_timestamp,
            bid.zone,
            bid.side,
            bid.limit_price_eur_mwh,
            bid.agent_id,
            bid.asset_id,
        ),
    )


def _bid_rejection_reason(
    bid: Bid,
    *,
    timestamp: datetime,
    zone: str,
    direction: Direction,
    marginal_price: float,
) -> str | None:
    if bid.utc_timestamp != timestamp:
        return "wrong_tick"
    if bid.zone != zone:
        return "wrong_zone"
    if direction == "neutral":
        return "no_activation_need"
    if bid.side != direction:
        return "opposite_activation_direction"
    if bid.limit_price_eur_mwh - marginal_price > 1e-9:
        return "limit_price_not_crossed"
    return None


def _decision(
    bid: Bid,
    accepted: bool,
    reason_code: str | None,
    settlement_eur: float,
    *,
    quantity_mwh: float | None = None,
) -> ClearingBidDecision:
    return ClearingBidDecision(
        agent_id=bid.agent_id,
        asset_id=bid.asset_id,
        zone=bid.zone,
        side=bid.side,
        quantity_mwh=round(bid.quantity_mwh if quantity_mwh is None else quantity_mwh, 6),
        limit_price_eur_mwh=round(bid.limit_price_eur_mwh, 6),
        accepted=accepted,
        reason_code=reason_code,
        settlement_eur=round(settlement_eur, 6),
    )


def _settlement(
    *,
    side: str,
    quantity_mwh: float,
    spot_price_eur_mwh: float,
    clearing_price_eur_mwh: float,
) -> float:
    if side == "up":
        return quantity_mwh * (clearing_price_eur_mwh - spot_price_eur_mwh)
    return quantity_mwh * (spot_price_eur_mwh - clearing_price_eur_mwh)


def _anchored_counterfactual_price(
    *,
    direction: Direction,
    historical_price: float,
    spot_price: float,
    accepted_volume_mwh: float,
    slope_eur_per_mwh: float,
) -> float:
    if direction == "neutral":
        return spot_price
    if direction == "up":
        return max(spot_price, historical_price - slope_eur_per_mwh * accepted_volume_mwh)
    return min(spot_price, historical_price + slope_eur_per_mwh * accepted_volume_mwh)


def _hash_payload(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "result_hash"}
    encoded = json.dumps(clean, default=str, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
