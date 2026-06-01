from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from packages.config import load_project_env


class PyPSAScenarioError(RuntimeError):
    """Raised when PyPSA scenario input is missing or malformed."""


@dataclass(frozen=True)
class PyPSAEurConfig:
    pypsa_eur_dir: Path
    solver_name: str = "highs"

    @classmethod
    def from_env(cls) -> "PyPSAEurConfig":
        load_project_env()
        heimdall_data_dir = Path(
            os.environ.get("HEIMDALL_DATA_DIR", "").strip() or "data"
        )
        raw_dir = (
            os.environ.get("HEIMDALL_PYPSA_EUR_DIR", "").strip()
            or os.environ.get("PYPSA_EUR_DIR", "").strip()
        )
        solver_name = (
            os.environ.get("HEIMDALL_PYPSA_SOLVER", "").strip()
            or os.environ.get("PYPSA_SOLVER", "").strip()
            or "highs"
        )
        path = Path(raw_dir) if raw_dir else heimdall_data_dir / "cache" / "pypsa-eur"
        return cls(pypsa_eur_dir=path.expanduser().resolve(), solver_name=solver_name)

    def validate_existing(self) -> "PyPSAEurConfig":
        resolved = self.pypsa_eur_dir.expanduser().resolve()
        if not resolved.exists():
            raise PyPSAScenarioError(f"PYPSA_EUR_DIR does not exist: {resolved}")
        return PyPSAEurConfig(pypsa_eur_dir=resolved, solver_name=self.solver_name)
