from pathlib import Path

import pytest

from packages.data import DataConfig, DataSourceError


REQUIRED_ENV_KEYS = [
    "TRADERS_TRINITY_DATA_DIR",
    "ENTSOE_API_KEY",
    "NORDPOOL_API_KEY",
    "SERPER_API_KEY",
    "HF_TOKEN",
    "HEIMDALL_PYPSA_EUR_DIR",
    "HEIMDALL_PYPSA_SOLVER",
    "HEIMDALL_DATA_DIR",
    "JAO_API_KEY",
]

REQUIRED_EMPTY_ENV_KEYS = [
    "TRADERS_TRINITY_DATA_DIR",
    "ENTSOE_API_KEY",
    "NORDPOOL_API_KEY",
    "SERPER_API_KEY",
    "HF_TOKEN",
    "HEIMDALL_PYPSA_EUR_DIR",
    "HEIMDALL_DATA_DIR",
    "JAO_API_KEY",
]


def test_env_file_is_gitignored() -> None:
    gitignore = Path(".gitignore").read_text()

    assert ".env" in gitignore.splitlines()
    assert ".env.local" in gitignore.splitlines()
    assert ".env.*.local" in gitignore.splitlines()


def test_env_example_contains_empty_required_keys() -> None:
    lines = Path(".env.example").read_text().splitlines()
    values = dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))

    assert set(REQUIRED_ENV_KEYS).issubset(values)
    assert all(values[key] == "" for key in REQUIRED_EMPTY_ENV_KEYS)
    assert values["HEIMDALL_PYPSA_SOLVER"] == "highs"
    assert values["JAO_BASE_URL"] == "https://publicationtool.jao.eu"
    assert values["JAO_ENABLE_LIVE_FETCH"] == "false"
    assert values["JAO_CACHE_DIR"] == "data/raw/jao"


def test_loader_resolves_traders_trinity_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("TRADERS_TRINITY_DATA_DIR", str(data_dir))

    config = DataConfig.from_env()

    assert config.traders_trinity_data_dir == data_dir.resolve()


def test_loader_raises_structured_error_for_missing_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRADERS_TRINITY_DATA_DIR", str(tmp_path / "missing"))

    with pytest.raises(DataSourceError, match="TRADERS_TRINITY_DATA_DIR"):
        DataConfig.from_env()
