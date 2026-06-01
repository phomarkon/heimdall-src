import os
from pathlib import Path

from packages.config import env_status, load_project_env
from packages.data import DataConfig
from packages.pypsa_adapter import PyPSAEurConfig


def test_project_env_loads_dotenv_without_overriding_shell(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TRADERS_TRINITY_DATA_DIR=/tmp/from-dotenv",
                "HEIMDALL_PYPSA_EUR_DIR=/tmp/pypsa-eur",
                "HEIMDALL_PYPSA_SOLVER=highs",
                "ENTSOE_API_KEY=secret-value",
                "JAO_API_KEY=jao-secret",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADERS_TRINITY_DATA_DIR", "/tmp/from-shell")
    monkeypatch.delenv("HEIMDALL_PYPSA_EUR_DIR", raising=False)

    loaded = load_project_env(tmp_path)

    assert loaded == env_file
    assert os.environ["TRADERS_TRINITY_DATA_DIR"] == "/tmp/from-shell"
    assert os.environ["HEIMDALL_PYPSA_EUR_DIR"] == "/tmp/pypsa-eur"
    assert env_status(["ENTSOE_API_KEY", "NORDPOOL_API_KEY", "JAO_API_KEY"]) == {
        "ENTSOE_API_KEY": "SET",
        "NORDPOOL_API_KEY": "EMPTY",
        "JAO_API_KEY": "SET",
    }


def test_data_config_loads_traders_path_from_dotenv(
    tmp_path: Path, monkeypatch
) -> None:
    data_dir = tmp_path / "trinity" / "data"
    data_dir.mkdir(parents=True)
    (tmp_path / ".env").write_text(
        f"TRADERS_TRINITY_DATA_DIR={data_dir}\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRADERS_TRINITY_DATA_DIR", raising=False)

    config = DataConfig.from_env()

    assert config.traders_trinity_data_dir == data_dir.resolve()


def test_pypsa_config_defaults_to_ignored_cache_when_dotenv_key_empty(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".env").write_text(
        "HEIMDALL_PYPSA_EUR_DIR=\nHEIMDALL_PYPSA_SOLVER=\nHEIMDALL_DATA_DIR=data\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEIMDALL_PYPSA_EUR_DIR", raising=False)
    monkeypatch.delenv("HEIMDALL_PYPSA_SOLVER", raising=False)
    monkeypatch.delenv("HEIMDALL_DATA_DIR", raising=False)

    config = PyPSAEurConfig.from_env()

    assert config.pypsa_eur_dir == (tmp_path / "data/cache/pypsa-eur").resolve()
    assert config.solver_name == "highs"
