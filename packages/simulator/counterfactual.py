from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import pandas as pd


@dataclass(frozen=True)
class PriceImpactOutcome:
    baseline_imbalance_price_eur_mwh: float
    counterfactual_imbalance_price_eur_mwh: float
    baseline_activation_volume_mwh: float
    counterfactual_activation_volume_mwh: float
    result_hash: str


@dataclass(frozen=True)
class PriceImpactModel:
    price_impact_eur_per_mwh_per_mwh: float

    @classmethod
    def from_eds_rows(
        cls,
        rows: pd.DataFrame,
        *,
        fallback_price_impact_eur_per_mwh_per_mwh: float = 0.0,
    ) -> "PriceImpactModel":
        required = {
            "spot_price_eur_mwh",
            "imbalance_price_eur_mwh",
            "satisfied_demand_mw",
        }
        missing = required.difference(rows.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"EDS rows missing required columns: {missing_columns}")

        activation_mwh = pd.to_numeric(rows["satisfied_demand_mw"]).abs() * 0.25
        spread = pd.to_numeric(rows["imbalance_price_eur_mwh"]) - pd.to_numeric(
            rows["spot_price_eur_mwh"]
        )
        valid = pd.DataFrame({"activation": activation_mwh, "spread": spread}).dropna()
        if len(valid) < 2:
            return cls(fallback_price_impact_eur_per_mwh_per_mwh)

        activation_centered = valid["activation"] - valid["activation"].mean()
        spread_centered = valid["spread"] - valid["spread"].mean()
        denominator = float((activation_centered**2).sum())
        if denominator == 0.0:
            return cls(fallback_price_impact_eur_per_mwh_per_mwh)

        slope = float((activation_centered * spread_centered).sum() / denominator)
        return cls(round(abs(slope), 6))

    def apply(
        self,
        *,
        spot_price_eur_mwh: float,
        imbalance_price_eur_mwh: float,
        activation_volume_mwh: float,
        entrant_shift_mwh: float,
    ) -> PriceImpactOutcome:
        counterfactual_volume = activation_volume_mwh - entrant_shift_mwh
        baseline_spread = imbalance_price_eur_mwh - spot_price_eur_mwh
        counterfactual_spread = baseline_spread - (
            entrant_shift_mwh * self.price_impact_eur_per_mwh_per_mwh
        )
        counterfactual_price = spot_price_eur_mwh + counterfactual_spread
        outcome = PriceImpactOutcome(
            baseline_imbalance_price_eur_mwh=round(imbalance_price_eur_mwh, 6),
            counterfactual_imbalance_price_eur_mwh=round(counterfactual_price, 6),
            baseline_activation_volume_mwh=round(activation_volume_mwh, 6),
            counterfactual_activation_volume_mwh=round(counterfactual_volume, 6),
            result_hash="",
        )
        return PriceImpactOutcome(
            **{**asdict(outcome), "result_hash": _hash_payload(asdict(outcome))}
        )


@dataclass(frozen=True)
class CounterfactualEntrant:
    asset_id: str
    zone: str
    p2h_capacity_mw: float
    thermal_storage_mwh: float
    dispatch_mwh: float


@dataclass(frozen=True)
class CounterfactualComparison:
    baseline_price_eur_mwh: float
    counterfactual_price_eur_mwh: float
    price_delta_eur_mwh: float
    baseline_activation_volume_mwh: float
    counterfactual_activation_volume_mwh: float
    activation_delta_mwh: float
    result_hash: str


def compare_counterfactual(
    model: PriceImpactModel,
    entrant: CounterfactualEntrant,
    *,
    spot_price_eur_mwh: float,
    imbalance_price_eur_mwh: float,
    activation_volume_mwh: float,
) -> CounterfactualComparison:
    shift_mwh = min(entrant.dispatch_mwh, entrant.p2h_capacity_mw * 0.25)
    outcome = model.apply(
        spot_price_eur_mwh=spot_price_eur_mwh,
        imbalance_price_eur_mwh=imbalance_price_eur_mwh,
        activation_volume_mwh=activation_volume_mwh,
        entrant_shift_mwh=shift_mwh,
    )
    comparison = CounterfactualComparison(
        baseline_price_eur_mwh=outcome.baseline_imbalance_price_eur_mwh,
        counterfactual_price_eur_mwh=outcome.counterfactual_imbalance_price_eur_mwh,
        price_delta_eur_mwh=round(
            outcome.counterfactual_imbalance_price_eur_mwh
            - outcome.baseline_imbalance_price_eur_mwh,
            6,
        ),
        baseline_activation_volume_mwh=outcome.baseline_activation_volume_mwh,
        counterfactual_activation_volume_mwh=outcome.counterfactual_activation_volume_mwh,
        activation_delta_mwh=round(
            outcome.counterfactual_activation_volume_mwh
            - outcome.baseline_activation_volume_mwh,
            6,
        ),
        result_hash="",
    )
    return CounterfactualComparison(
        **{**asdict(comparison), "result_hash": _hash_payload(asdict(comparison))}
    )


def _hash_payload(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "result_hash"}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
