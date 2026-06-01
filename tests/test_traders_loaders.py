from pathlib import Path

import pandas as pd
import pytest

from packages.data import DataConfig, DataSourceError, load_traders_table


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture()
def traders_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "traders" / "data"
    processed = data_dir / "processed"
    timestamp = "2026-04-30T00:00:00Z"

    _write_parquet(
        processed / "prices.parquet",
        [
            {"utc_timestamp": timestamp, "zone": "DK1", "price_eur_mwh": 91.0},
            {"utc_timestamp": timestamp, "zone": "DK2", "price_eur_mwh": 92.0},
        ],
    )
    _write_parquet(
        processed / "load.parquet",
        [
            {"utc_timestamp": timestamp, "zone": "DK1", "load_mw": 2100.0},
            {"utc_timestamp": timestamp, "zone": "DK2", "load_mw": 1600.0},
        ],
    )
    _write_parquet(
        processed / "generation.parquet",
        [
            {"utc_timestamp": timestamp, "zone": "DK1", "Wind": 1200.0},
            {"utc_timestamp": timestamp, "zone": "DK2", "Wind": 900.0},
        ],
    )
    _write_parquet(
        processed / "flows.parquet",
        [
            {
                "utc_timestamp": timestamp,
                "from_zone": "DK1",
                "to_zone": "DK2",
                "flow_mw": 100.0,
            }
        ],
    )
    _write_parquet(
        processed / "weather.parquet",
        [
            {"utc_timestamp": timestamp, "zone": "DK1", "temperature_2m": 7.0},
            {"utc_timestamp": timestamp, "zone": "DK2", "temperature_2m": 8.0},
        ],
    )
    return data_dir


@pytest.mark.parametrize(
    ("table", "expected_columns"),
    [
        ("prices", {"utc_timestamp", "zone", "price_eur_mwh"}),
        ("load", {"utc_timestamp", "zone", "load_mw"}),
        ("generation", {"utc_timestamp", "zone"}),
        ("flows", {"utc_timestamp", "from_zone", "to_zone", "flow_mw"}),
        ("weather", {"utc_timestamp", "zone", "temperature_2m"}),
    ],
)
def test_load_traders_tables_have_expected_columns_and_utc_timestamps(
    traders_data_dir: Path, table: str, expected_columns: set[str]
) -> None:
    config = DataConfig(traders_trinity_data_dir=traders_data_dir)

    frame = load_traders_table(config, table)

    assert expected_columns.issubset(frame.columns)
    assert str(frame["utc_timestamp"].dt.tz) == "UTC"


@pytest.mark.parametrize("table", ["prices", "load", "generation", "weather"])
def test_zone_tables_include_dk1_and_dk2(traders_data_dir: Path, table: str) -> None:
    config = DataConfig(traders_trinity_data_dir=traders_data_dir)

    frame = load_traders_table(config, table)

    assert {"DK1", "DK2"}.issubset(set(frame["zone"]))


def test_missing_traders_table_raises_structured_error(traders_data_dir: Path) -> None:
    (traders_data_dir / "processed" / "prices.parquet").unlink()
    config = DataConfig(traders_trinity_data_dir=traders_data_dir)

    with pytest.raises(DataSourceError, match="prices.parquet"):
        load_traders_table(config, "prices")
