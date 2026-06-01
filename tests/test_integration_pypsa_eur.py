import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_HEIMDALL_PYPSA_EUR") != "1",
    reason="set RUN_HEIMDALL_PYPSA_EUR=1 to run PyPSA-Eur smoke tests",
)
def test_pypsa_eur_bootstrap_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "research/tools/pypsa/bootstrap_pypsa_eur.py",
            "--dry-run",
            "--ref",
            "v2026.02.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "PyPSA-Eur" in result.stdout
    assert "v2026.02.0" in result.stdout
