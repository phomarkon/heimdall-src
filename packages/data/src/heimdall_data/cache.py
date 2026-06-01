from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CachedFrame:
    frame: pd.DataFrame
    path: Path
    fresh: bool
    source: str
    fetched_at_utc: str | None


def data_cache_dir() -> Path:
    root = os.environ.get("HEIMDALL_DATA_DIR", "").strip()
    base = Path(root) if root else Path("data")
    return base / "cache"


def read_cached_frame(key: str, *, cache_dir: Path | None = None) -> CachedFrame | None:
    path = (cache_dir or data_cache_dir()) / f"{key}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    meta = _read_meta(path)
    return CachedFrame(
        frame=frame,
        path=path,
        fresh=False,
        source=str(meta.get("source", "cache")),
        fetched_at_utc=meta.get("fetched_at_utc"),
    )


def write_cached_frame(
    key: str,
    frame: pd.DataFrame,
    *,
    source: str,
    cache_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> CachedFrame:
    path = (cache_dir or data_cache_dir()) / f"{key}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    fetched_at = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    meta = {"source": source, "fetched_at_utc": fetched_at, **(metadata or {})}
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CachedFrame(frame=frame, path=path, fresh=True, source=source, fetched_at_utc=fetched_at)


def _read_meta(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))
