"""Shared utilities for the run-view service.

Pure data classes, I/O helpers, and type-coercion functions used by both
the disk-based adapter and the Postgres-backed database layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    trace_path: Path
    manifest_path: Path | None
    trace_sha256: str
    total_steps: int
    status: str = "completed"
    trace_rows: list[dict[str, Any]] | None = None
    summary_json: dict[str, Any] | None = None
    bid_evaluation_rows: list[dict[str, Any]] | None = None
    config_json: dict[str, Any] | None = None
    source_run_dir: str | None = None


@dataclass(frozen=True)
class RunContext:
    summary: dict[str, Any]
    bid_rows_by_step_agent: dict[tuple[int, str], dict[str, Any]]
    bid_rows_by_step: dict[int, list[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------


def _optional_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return round(float(value), 6)
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _int_like(value: Any, fallback: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return fallback


def _float(value: Any, default: float) -> float:
    return float(value) if isinstance(value, int | float) else default


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def _parse_timestamp(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _parse_dt(value: Any) -> datetime:
    """Strict variant that raises on un-parseable input."""
    result = _parse_timestamp(value)
    if result is None:
        raise ValueError(f"Cannot parse timestamp: {value!r}")
    return result


def _iso_z(value: Any) -> str:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return str(value)
    return parsed.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def default_repo_root() -> Path:
    env_root = os.getenv("HEIMDALL_RUN_VIEW_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[4]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_first_jsonl(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def _scan_trace_metadata(path: Path) -> tuple[dict[str, Any], int] | None:
    first: dict[str, Any] | None = None
    max_step = -1
    step_values: set[int] = set()
    timestamp_values: set[str] = set()
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for _index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if first is None:
                first = row
            row_count += 1
            raw_step = row.get("step")
            if isinstance(raw_step, int):
                step_values.add(raw_step)
                max_step = max(max_step, raw_step)
            elif isinstance(raw_step, float) and raw_step.is_integer():
                step = int(raw_step)
                step_values.add(step)
                max_step = max(max_step, step)
            timestamp = (
                row.get("timestamp") or row.get("utc_timestamp") or row.get("delivery_quarter")
            )
            if timestamp:
                timestamp_values.add(str(timestamp))
    if first is None:
        return None
    if step_values:
        return first, max(max_step + 1, len(step_values))
    if timestamp_values:
        return first, len(timestamp_values)
    return first, row_count


def _normalize_parquet_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            normalized[key] = value.astimezone(UTC).isoformat().replace("+00:00", "Z")
        elif isinstance(value, float) and math.isnan(value):
            normalized[key] = None
        else:
            normalized[key] = value
    return normalized


# ---------------------------------------------------------------------------
# Run artifact readers
# ---------------------------------------------------------------------------


def _read_run_summary(record: RunRecord, manifest: dict[str, Any]) -> dict[str, Any]:
    if record.summary_json is not None:
        return record.summary_json
    candidates = []
    artifact = (manifest.get("artifacts") or {}).get("run_summary")
    if artifact:
        candidates.append(record.trace_path.parents[3] / artifact)
        candidates.append(default_repo_root() / artifact)
    candidates.append(default_repo_root() / "research" / "llm" / "evaluations" / record.run_id / "run_summary.json")
    for path in candidates:
        if path.exists():
            return _read_json(path)
    return {}


def _read_run_summary_for_root(root: Path, record: RunRecord) -> dict[str, Any]:
    if record.summary_json is not None:
        return record.summary_json
    candidates = [
        root / "research" / "llm" / "evaluations" / record.run_id / "run_summary.json",
        record.trace_path.parent / "run_summary.json",
    ]
    for path in candidates:
        if path.exists():
            return _read_json(path)
    return {}


def _read_bid_evaluations(record: RunRecord, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if record.bid_evaluation_rows is not None:
        return record.bid_evaluation_rows
    artifact = (manifest.get("artifacts") or {}).get("bid_evaluations")
    candidates = []
    if artifact:
        candidates.append(default_repo_root() / artifact)
    candidates.append(
        default_repo_root() / "research" / "llm" / "evaluations" / record.run_id / "bid_evaluations.parquet"
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return []
    try:
        import pyarrow.parquet as pq
    except Exception:
        return []
    table = pq.read_table(path)
    return [_normalize_parquet_row(row) for row in table.to_pylist()]


def _load_run_context(record: RunRecord, manifest: dict[str, Any]) -> RunContext:
    summary = _read_run_summary(record, manifest)
    bid_rows = _read_bid_evaluations(record, manifest)
    by_step_agent: dict[tuple[int, str], dict[str, Any]] = {}
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in bid_rows:
        step = int(row.get("step", 0))
        agent_id = str(row.get("agent_id") or "")
        by_step.setdefault(step, []).append(row)
        if agent_id:
            by_step_agent[(step, agent_id)] = row
    return RunContext(
        summary=summary, bid_rows_by_step_agent=by_step_agent, bid_rows_by_step=by_step
    )
