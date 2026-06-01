from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from packages.config import load_project_env


DEFAULT_TRADERS_TRINITY_DATA_DIR = Path(
    "/Users/timadam/Code/the-traders-trinity-ml-debate/data"
)


class DataSourceError(RuntimeError):
    """Raised when a configured data source is missing or malformed."""


@dataclass(frozen=True)
class DataConfig:
    traders_trinity_data_dir: Path

    def __post_init__(self) -> None:
        resolved = self.traders_trinity_data_dir.expanduser().resolve()
        if not resolved.exists():
            raise DataSourceError(
                f"TRADERS_TRINITY_DATA_DIR does not exist: {resolved}"
            )
        object.__setattr__(self, "traders_trinity_data_dir", resolved)

    @classmethod
    def from_env(cls) -> "DataConfig":
        load_project_env()
        raw_path = os.environ.get("TRADERS_TRINITY_DATA_DIR", "").strip()
        path = Path(raw_path) if raw_path else DEFAULT_TRADERS_TRINITY_DATA_DIR
        return cls(traders_trinity_data_dir=path)
