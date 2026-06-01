from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import DataConfig, DataSourceError


TRADERS_TABLES = {
    "prices": ("processed/prices.parquet", {"utc_timestamp", "zone", "price_eur_mwh"}),
    "load": ("processed/load.parquet", {"utc_timestamp", "zone", "load_mw"}),
    "generation": ("processed/generation.parquet", {"utc_timestamp", "zone"}),
    "flows": ("processed/flows.parquet", {"utc_timestamp", "from_zone", "to_zone", "flow_mw"}),
    "weather": ("processed/weather.parquet", {"utc_timestamp", "zone"}),
}


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataSourceError(f"{source} missing required columns: {missing}")


def _normalize_utc_column(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if "utc_timestamp" not in frame.columns:
        raise DataSourceError(f"{source} missing required column: utc_timestamp")

    normalized = frame.copy()
    normalized["utc_timestamp"] = pd.to_datetime(
        normalized["utc_timestamp"], utc=True, errors="coerce"
    )
    if normalized["utc_timestamp"].isna().any():
        raise DataSourceError(f"{source} contains invalid utc_timestamp values")
    return normalized


def load_traders_table(config: DataConfig, table: str) -> pd.DataFrame:
    if table not in TRADERS_TABLES:
        raise DataSourceError(f"Unknown Trader's Trinity table: {table}")

    relative_path, required_columns = TRADERS_TABLES[table]
    path = config.traders_trinity_data_dir / relative_path
    if not path.exists():
        raise DataSourceError(f"Trader's Trinity table not found: {path}")

    frame = pd.read_parquet(path)
    _require_columns(frame, required_columns, str(path))
    return _normalize_utc_column(frame, str(path))
