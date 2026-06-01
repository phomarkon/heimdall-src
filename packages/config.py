from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def find_project_env(start: Path | None = None) -> Path | None:
    cursor = (start or Path.cwd()).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for directory in [cursor, *cursor.parents]:
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None


def load_project_env(start: Path | None = None) -> Path | None:
    env_path = find_project_env(start)
    if env_path is None:
        return None
    load_dotenv(env_path, override=False)
    return env_path


def env_status(keys: list[str]) -> dict[str, str]:
    return {key: "SET" if os.environ.get(key, "").strip() else "EMPTY" for key in keys}
