from pathlib import Path
import os
import subprocess
import sys

import pytest

from packages.pypsa_adapter import PyPSAEurConfig, PyPSAScenarioError
from tools.pypsa.bootstrap_pypsa_eur import resolve_target_dir


def test_missing_pypsa_eur_path_raises_structured_error(tmp_path: Path) -> None:
    config = PyPSAEurConfig(pypsa_eur_dir=tmp_path / "missing")

    with pytest.raises(PyPSAScenarioError, match="PYPSA_EUR_DIR"):
        config.validate_existing()


def test_bootstrap_resolves_target_dir_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "cache" / "pypsa-eur"
    (tmp_path / ".env").write_text(
        f"HEIMDALL_PYPSA_EUR_DIR={target}\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEIMDALL_PYPSA_EUR_DIR", raising=False)

    assert resolve_target_dir(None) == target.resolve()


def test_pypsa_import_does_not_treat_heimdall_env_as_pypsa_options() -> None:
    env = os.environ.copy()
    env["HEIMDALL_PYPSA_EUR_DIR"] = "data/cache/pypsa-eur"
    env["HEIMDALL_PYPSA_SOLVER"] = "highs"
    env.pop("PYPSA_EUR_DIR", None)
    env.pop("PYPSA_SOLVER", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from packages.pypsa_adapter import build_tiny_dk_network; build_tiny_dk_network()",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Unknown option" not in result.stderr
