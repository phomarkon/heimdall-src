import pytest
import pandas as pd

from packages.simulator.counterfactual import (
    CounterfactualEntrant,
    PriceImpactModel,
    compare_counterfactual,
)


def test_price_impact_model_is_deterministic_and_directional() -> None:
    model = PriceImpactModel(price_impact_eur_per_mwh_per_mwh=2.5)

    baseline = model.apply(
        spot_price_eur_mwh=50.0,
        imbalance_price_eur_mwh=80.0,
        activation_volume_mwh=10.0,
        entrant_shift_mwh=0.0,
    )
    counterfactual = model.apply(
        spot_price_eur_mwh=50.0,
        imbalance_price_eur_mwh=80.0,
        activation_volume_mwh=10.0,
        entrant_shift_mwh=4.0,
    )
    repeat = model.apply(
        spot_price_eur_mwh=50.0,
        imbalance_price_eur_mwh=80.0,
        activation_volume_mwh=10.0,
        entrant_shift_mwh=4.0,
    )

    assert baseline.counterfactual_imbalance_price_eur_mwh == 80.0
    assert counterfactual.counterfactual_activation_volume_mwh == 6.0
    assert counterfactual.counterfactual_imbalance_price_eur_mwh == 70.0
    assert counterfactual.result_hash == repeat.result_hash


def test_counterfactual_entrant_changes_price_and_activation_deterministically() -> None:
    model = PriceImpactModel(price_impact_eur_per_mwh_per_mwh=2.5)
    entrant = CounterfactualEntrant(
        asset_id="DK1",
        zone="DK1",
        p2h_capacity_mw=50.0,
        thermal_storage_mwh=100.0,
        dispatch_mwh=4.0,
    )

    result = compare_counterfactual(
        model,
        entrant,
        spot_price_eur_mwh=50.0,
        imbalance_price_eur_mwh=80.0,
        activation_volume_mwh=10.0,
    )

    assert result.baseline_price_eur_mwh == 80.0
    assert result.counterfactual_price_eur_mwh == pytest.approx(70.0)
    assert result.price_delta_eur_mwh == pytest.approx(-10.0)
    assert result.activation_delta_mwh == pytest.approx(-4.0)
    assert result.result_hash


def test_price_impact_model_can_be_calibrated_from_eds_shaped_rows() -> None:
    rows = pd.DataFrame(
        {
            "spot_price_eur_mwh": [50.0, 50.0, 50.0],
            "imbalance_price_eur_mwh": [50.0, 60.0, 70.0],
            "satisfied_demand_mw": [0.0, 16.0, 32.0],
        }
    )

    model = PriceImpactModel.from_eds_rows(rows)

    assert model.price_impact_eur_per_mwh_per_mwh == pytest.approx(2.5)
