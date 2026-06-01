from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_hash(columns: list[str]) -> str:
    payload = json.dumps(columns, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(
    *,
    artifact_path: Path,
    source_url: str,
    dataset: str,
    window_start_utc: str,
    window_end_utc: str,
    row_count: int,
    schema_columns: list[str],
) -> Path:
    artifact = artifact_path.resolve()
    manifest_path = artifact.with_suffix(f"{artifact.suffix}.manifest.json")
    manifest = {
        "artifact_path": str(artifact),
        "created_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "dataset": dataset,
        "file_sha256": file_sha256(artifact),
        "row_count": row_count,
        "schema_columns": schema_columns,
        "schema_hash": _schema_hash(schema_columns),
        "source_url": source_url,
        "window_end_utc": window_end_utc,
        "window_start_utc": window_start_utc,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path
