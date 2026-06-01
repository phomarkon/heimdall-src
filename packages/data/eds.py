from __future__ import annotations

from typing import Any

import pandas as pd
import requests


EDS_DATASET_URL = "https://api.energidataservice.dk/dataset"

IMBALANCE_REQUIRED_COLUMNS = {
    "TimeUTC",
    "PriceArea",
    "SatisfiedDemand",
    "ImbalancePriceEUR",
    "SpotPriceEUR",
    "mFRRMarginalPriceUpEUR",
    "mFRRMarginalPriceDownEUR",
}

DAY_AHEAD_REQUIRED_COLUMNS = {
    "TimeUTC",
    "PriceArea",
    "DayAheadPriceEUR",
}


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required EDS columns: {missing}")


def _canonical_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError("EDS records contain invalid TimeUTC values")
    return parsed


def _numeric(series: pd.Series, column: str) -> pd.Series:
    originally_missing = series.isna()
    parsed = pd.to_numeric(series, errors="coerce")
    invalid = parsed.isna() & ~originally_missing
    if invalid.any():
        raise ValueError(f"EDS records contain invalid numeric values in {column}")
    return parsed


def normalize_eds_imbalance_price(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, IMBALANCE_REQUIRED_COLUMNS)
    normalized = pd.DataFrame(
        {
            "utc_timestamp": _canonical_timestamp(frame["TimeUTC"]),
            "zone": frame["PriceArea"].astype(str),
            "satisfied_demand_mw": _numeric(frame["SatisfiedDemand"], "SatisfiedDemand"),
            "imbalance_price_eur_mwh": _numeric(
                frame["ImbalancePriceEUR"], "ImbalancePriceEUR"
            ),
            "spot_price_eur_mwh": _numeric(frame["SpotPriceEUR"], "SpotPriceEUR"),
            "mfrr_marginal_price_up_eur_mwh": _numeric(
                frame["mFRRMarginalPriceUpEUR"], "mFRRMarginalPriceUpEUR"
            ),
            "mfrr_marginal_price_down_eur_mwh": _numeric(
                frame["mFRRMarginalPriceDownEUR"], "mFRRMarginalPriceDownEUR"
            ),
        }
    )
    numeric_columns = [
        "satisfied_demand_mw",
        "imbalance_price_eur_mwh",
        "spot_price_eur_mwh",
        "mfrr_marginal_price_up_eur_mwh",
        "mfrr_marginal_price_down_eur_mwh",
    ]
    normalized = normalized.dropna(subset=numeric_columns).copy()
    normalized[numeric_columns] = normalized[numeric_columns].astype(float)
    return normalized.sort_values(["utc_timestamp", "zone"]).reset_index(drop=True)


def normalize_eds_day_ahead_prices(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, DAY_AHEAD_REQUIRED_COLUMNS)
    normalized = pd.DataFrame(
        {
            "utc_timestamp": _canonical_timestamp(frame["TimeUTC"]),
            "zone": frame["PriceArea"].astype(str),
            "day_ahead_price_eur_mwh": _numeric(
                frame["DayAheadPriceEUR"], "DayAheadPriceEUR"
            ),
        }
    )
    return normalized.sort_values(["utc_timestamp", "zone"]).reset_index(drop=True)


def fetch_eds_dataset(
    dataset: str,
    *,
    start: str,
    end: str,
    price_areas: list[str] | None = None,
    timeout_seconds: int = 120,
) -> pd.DataFrame:
    params: dict[str, Any] = {
        "start": start,
        "end": end,
        "sort": "TimeUTC asc",
        "limit": 0,
    }
    if price_areas:
        areas = ",".join(f'"{area}"' for area in price_areas)
        params["filter"] = f'{{"PriceArea":[{areas}]}}'

    response = requests.get(
        f"{EDS_DATASET_URL}/{dataset}", params=params, timeout=timeout_seconds
    )
    response.raise_for_status()
    payload = response.json()
    return pd.DataFrame(payload.get("records", []))
