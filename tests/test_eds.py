import pandas as pd
import pytest

from packages.data import normalize_eds_imbalance_price


IMBALANCE_RECORDS = [
    {
        "TimeUTC": "2025-03-04T22:30:00",
        "PriceArea": "DK1",
        "SatisfiedDemand": 233.0,
        "ImbalancePriceEUR": 121.82,
        "SpotPriceEUR": 20.69,
        "mFRRMarginalPriceUpEUR": 121.82,
        "mFRRMarginalPriceDownEUR": 20.69,
    },
    {
        "TimeUTC": "2025-03-04T22:30:00",
        "PriceArea": "DK2",
        "SatisfiedDemand": -261.0,
        "ImbalancePriceEUR": -30.0,
        "SpotPriceEUR": 20.10,
        "mFRRMarginalPriceUpEUR": 20.10,
        "mFRRMarginalPriceDownEUR": -30.0,
    },
]


def test_imbalance_price_normalization_requires_core_columns() -> None:
    raw = pd.DataFrame(IMBALANCE_RECORDS).drop(columns=["SatisfiedDemand"])

    with pytest.raises(ValueError, match="SatisfiedDemand"):
        normalize_eds_imbalance_price(raw)


def test_imbalance_price_normalizes_to_canonical_15_minute_rows() -> None:
    normalized = normalize_eds_imbalance_price(pd.DataFrame(IMBALANCE_RECORDS))

    assert list(normalized.columns) == [
        "utc_timestamp",
        "zone",
        "satisfied_demand_mw",
        "imbalance_price_eur_mwh",
        "spot_price_eur_mwh",
        "mfrr_marginal_price_up_eur_mwh",
        "mfrr_marginal_price_down_eur_mwh",
    ]
    assert set(normalized["zone"]) == {"DK1", "DK2"}
    assert str(normalized["utc_timestamp"].dt.tz) == "UTC"
    assert normalized["utc_timestamp"].diff().dropna().eq(pd.Timedelta(0)).all()
    assert normalized["imbalance_price_eur_mwh"].dtype.kind == "f"
