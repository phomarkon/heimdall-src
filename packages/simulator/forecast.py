from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
from typing import Literal

import pandas as pd

from .mfrr_engine import Direction


ForecastKind = Literal["baseline", "baseline_conformal", "oracle_actual", "external"]


@dataclass(frozen=True)
class ForecastSource:
    kind: ForecastKind | str
    window_start: str
    window_end: str | None = None
    method: str = "seasonal_rolling_quantile"


@dataclass(frozen=True)
class ForecastMarketState:
    delivery_timestamp: str
    zone: str
    issued_at: str
    activation_direction: Direction
    activation_volume_mwh: float
    spot_price_eur_mwh: float
    imbalance_price_lower_eur_mwh: float
    imbalance_price_median_eur_mwh: float
    imbalance_price_upper_eur_mwh: float
    mfrr_up_price_lower_eur_mwh: float
    mfrr_up_price_median_eur_mwh: float
    mfrr_up_price_upper_eur_mwh: float
    mfrr_down_price_lower_eur_mwh: float
    mfrr_down_price_median_eur_mwh: float
    mfrr_down_price_upper_eur_mwh: float
    source: ForecastSource
    result_hash: str = ""

    def __post_init__(self) -> None:
        _parse_utc(self.delivery_timestamp)
        _parse_utc(self.issued_at)
        if self.activation_volume_mwh < 0:
            raise ValueError("activation_volume_mwh must be non-negative")
        for lower, median, upper, name in [
            (
                self.imbalance_price_lower_eur_mwh,
                self.imbalance_price_median_eur_mwh,
                self.imbalance_price_upper_eur_mwh,
                "imbalance_price",
            ),
            (
                self.mfrr_up_price_lower_eur_mwh,
                self.mfrr_up_price_median_eur_mwh,
                self.mfrr_up_price_upper_eur_mwh,
                "mfrr_up_price",
            ),
            (
                self.mfrr_down_price_lower_eur_mwh,
                self.mfrr_down_price_median_eur_mwh,
                self.mfrr_down_price_upper_eur_mwh,
                "mfrr_down_price",
            ),
        ]:
            if lower > median or median > upper:
                raise ValueError(f"{name} interval must satisfy lower <= median <= upper")
        if not self.result_hash:
            object.__setattr__(self, "result_hash", _hash_payload(asdict(self)))

    def delivery_datetime(self) -> datetime:
        return _parse_utc(self.delivery_timestamp)

    def issued_datetime(self) -> datetime:
        return _parse_utc(self.issued_at)

    def interval_for_side(self, side: str) -> tuple[float, float]:
        if side == "up":
            return self.mfrr_up_price_lower_eur_mwh, self.mfrr_up_price_upper_eur_mwh
        if side == "down":
            return self.mfrr_down_price_lower_eur_mwh, self.mfrr_down_price_upper_eur_mwh
        raise ValueError(f"Unknown bid side: {side}")

    def to_market_row(self) -> dict[str, float | str]:
        signed_volume_mw = self.activation_volume_mwh / 0.25
        if self.activation_direction == "down":
            signed_volume_mw *= -1
        elif self.activation_direction == "neutral":
            signed_volume_mw = 0.0
        return {
            "utc_timestamp": self.delivery_timestamp,
            "zone": self.zone,
            "satisfied_demand_mw": round(signed_volume_mw, 6),
            "imbalance_price_eur_mwh": self.imbalance_price_median_eur_mwh,
            "spot_price_eur_mwh": self.spot_price_eur_mwh,
            "mfrr_marginal_price_up_eur_mwh": self.mfrr_up_price_median_eur_mwh,
            "mfrr_marginal_price_down_eur_mwh": self.mfrr_down_price_median_eur_mwh,
        }


class BaselineMFRRForecaster:
    def __init__(
        self,
        history: pd.DataFrame,
        *,
        lookback_days: int,
        alpha: float,
        calibration_adjustment_eur_mwh: float,
        window_start: str,
        window_end: str,
    ) -> None:
        self._history = history
        self._lookback_days = lookback_days
        self._alpha = alpha
        self._calibration_adjustment = calibration_adjustment_eur_mwh
        self._window_start = window_start
        self._window_end = window_end

    @classmethod
    def fit(
        cls,
        rows: pd.DataFrame,
        *,
        lookback_days: int = 28,
        calibration_fraction: float = 0.2,
        alpha: float = 0.1,
    ) -> "BaselineMFRRForecaster":
        history = _normalize_history(rows)
        if history.empty:
            raise ValueError("No forecast history available")
        split_index = max(1, int(len(history) * (1.0 - calibration_fraction)))
        train = history.iloc[:split_index]
        calibration = history.iloc[split_index:]
        adjustment = _conformal_adjustment(train, calibration, alpha)
        return cls(
            history,
            lookback_days=lookback_days,
            alpha=alpha,
            calibration_adjustment_eur_mwh=adjustment,
            window_start=_iso_z(history["utc_timestamp"].min()),
            window_end=_iso_z(history["utc_timestamp"].max()),
        )

    def forecast(
        self,
        *,
        delivery_timestamp: str,
        zone: str,
        issued_at: str,
    ) -> ForecastMarketState:
        delivery = _parse_utc(delivery_timestamp)
        issued = _parse_utc(issued_at)
        window_start = delivery - pd.Timedelta(days=self._lookback_days)
        candidates = self._history[
            (self._history["zone"] == zone)
            & (self._history["utc_timestamp"] < pd.Timestamp(issued))
            & (self._history["utc_timestamp"] >= pd.Timestamp(window_start))
            & (self._history["hour"] == delivery.hour)
            & (self._history["minute"] == delivery.minute)
        ]
        if candidates.empty:
            candidates = self._history[
                (self._history["zone"] == zone)
                & (self._history["utc_timestamp"] < pd.Timestamp(issued))
                & (self._history["utc_timestamp"] >= pd.Timestamp(window_start))
            ]
        if candidates.empty:
            raise ValueError(f"No forecast history for {zone} before {delivery_timestamp}")

        direction = _modal_direction(candidates)
        activation_volume = float(candidates["activation_volume_mwh"].median())
        spot = float(candidates["spot_price_eur_mwh"].median())
        imbalance_lower, imbalance_median, imbalance_upper = _interval(
            candidates["imbalance_price_eur_mwh"], self._calibration_adjustment
        )
        up_lower, up_median, up_upper = _interval(
            candidates["mfrr_marginal_price_up_eur_mwh"], self._calibration_adjustment
        )
        down_lower, down_median, down_upper = _interval(
            candidates["mfrr_marginal_price_down_eur_mwh"], self._calibration_adjustment
        )
        return ForecastMarketState(
            delivery_timestamp=_iso_z(delivery),
            zone=zone,
            issued_at=_iso_z(issued),
            activation_direction=direction,
            activation_volume_mwh=round(activation_volume, 6),
            spot_price_eur_mwh=round(spot, 6),
            imbalance_price_lower_eur_mwh=imbalance_lower,
            imbalance_price_median_eur_mwh=imbalance_median,
            imbalance_price_upper_eur_mwh=imbalance_upper,
            mfrr_up_price_lower_eur_mwh=up_lower,
            mfrr_up_price_median_eur_mwh=up_median,
            mfrr_up_price_upper_eur_mwh=up_upper,
            mfrr_down_price_lower_eur_mwh=down_lower,
            mfrr_down_price_median_eur_mwh=down_median,
            mfrr_down_price_upper_eur_mwh=down_upper,
            source=ForecastSource(
                kind="baseline_conformal",
                window_start=self._window_start,
                window_end=self._window_end,
            ),
        )


def _normalize_history(rows: pd.DataFrame) -> pd.DataFrame:
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
        raise ValueError(f"Forecast history missing required columns: {missing}")
    history = rows.copy()
    history["utc_timestamp"] = pd.to_datetime(
        history["utc_timestamp"], utc=True, errors="coerce"
    )
    numeric_columns = sorted(required - {"utc_timestamp", "zone"})
    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.dropna(subset=["utc_timestamp", "zone", *numeric_columns])
    history["activation_direction"] = history["satisfied_demand_mw"].map(
        _activation_direction
    )
    history["activation_volume_mwh"] = history["satisfied_demand_mw"].abs() * 0.25
    history["hour"] = history["utc_timestamp"].dt.hour
    history["minute"] = history["utc_timestamp"].dt.minute
    return history.sort_values(["utc_timestamp", "zone"]).reset_index(drop=True)


def _conformal_adjustment(
    train: pd.DataFrame, calibration: pd.DataFrame, alpha: float
) -> float:
    if train.empty or calibration.empty:
        return 0.0
    scores: list[float] = []
    for row in calibration.to_dict(orient="records"):
        candidates = train[
            (train["zone"] == row["zone"])
            & (train["hour"] == row["hour"])
            & (train["minute"] == row["minute"])
        ]
        if candidates.empty:
            candidates = train[train["zone"] == row["zone"]]
        if candidates.empty:
            continue
        lower, _, upper = _raw_quantiles(candidates["imbalance_price_eur_mwh"])
        target = float(row["imbalance_price_eur_mwh"])
        scores.append(max(lower - target, target - upper, 0.0))
    if not scores:
        return 0.0
    return round(float(pd.Series(scores).quantile(1.0 - alpha)), 6)


def _interval(series: pd.Series, adjustment: float) -> tuple[float, float, float]:
    lower, median, upper = _raw_quantiles(series)
    return (
        round(lower - adjustment, 6),
        round(median, 6),
        round(upper + adjustment, 6),
    )


def _raw_quantiles(series: pd.Series) -> tuple[float, float, float]:
    return (
        float(series.quantile(0.1)),
        float(series.quantile(0.5)),
        float(series.quantile(0.9)),
    )


def _modal_direction(rows: pd.DataFrame) -> Direction:
    counts = rows["activation_direction"].value_counts()
    if counts.empty:
        return "neutral"
    return str(counts.index[0])


def _activation_direction(value: float) -> Direction:
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "neutral"


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("forecast timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _iso_z(value) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _hash_payload(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "result_hash"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
