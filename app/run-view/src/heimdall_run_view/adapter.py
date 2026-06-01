"""Thin facade over the decomposed adapter sub-modules.

Public surface: RunCatalog, RunRecord, RunContext,
build_precomputed_run, build_agent_history.
"""

from __future__ import annotations

import json
from typing import Any

from heimdall_run_view._catalog import FOCAL_AGENT_ID, RunCatalog
from heimdall_run_view._forecaster import (
    _focal_baselines,
    _forecaster_leaderboard,
    _forecaster_summary,
)
from heimdall_run_view._priority import _priority_accuracy, _priority_signals
from heimdall_run_view._snapshot import _rows_by_step, build_snapshot
from heimdall_run_view._trace import _agent_history_record
from heimdall_run_view._utils import (
    RunContext,
    RunRecord,
    _load_run_context,
    _read_json,
    _read_jsonl,
)

__all__ = [
    "RunCatalog",
    "RunContext",
    "RunRecord",
    "build_agent_history",
    "build_precomputed_run",
]


def build_precomputed_run(record: RunRecord) -> dict[str, Any]:
    traces = record.trace_rows if record.trace_rows is not None else _read_jsonl(record.trace_path)
    manifest = _read_json(record.manifest_path) if record.manifest_path else {}
    context = _load_run_context(record, manifest)
    by_step = _rows_by_step(traces)
    total_steps = record.total_steps
    priority_by_step = _priority_signals(total_steps, by_step, context)
    priority_accuracy = _priority_accuracy(priority_by_step, context)
    snapshots = [
        build_snapshot(
            run_id=record.run_id,
            step=step,
            total_steps=total_steps,
            rows_by_step=by_step,
            manifest=manifest,
            trace_sha256=record.trace_sha256,
            context=context,
            priority_signal=priority_by_step.get(step),
        )
        for step in range(total_steps)
    ]
    return {
        "run_id": record.run_id,
        "total_steps": total_steps,
        "snapshots": snapshots,
        "market_series": [snapshot["market"] for snapshot in snapshots],
        "priority_accuracy": priority_accuracy,
        "forecaster_leaderboard": _forecaster_leaderboard(),
        "forecaster_summary": _forecaster_summary(
            run_id=record.run_id,
            total_steps=total_steps,
            rows_by_step=by_step,
            snapshots=snapshots,
            context=context,
        ),
        "focal_baselines": _focal_baselines(),
    }


def build_agent_history(record: RunRecord, agent_id: str) -> dict[str, Any]:
    manifest = _read_json(record.manifest_path) if record.manifest_path else {}
    context = _load_run_context(record, manifest)
    records: list[dict[str, Any]] = []
    if record.trace_rows is not None:
        for index, row in enumerate(record.trace_rows):
            row_agent_id = str(row.get("agent_id") or FOCAL_AGENT_ID)
            if row_agent_id == agent_id:
                records.append(_agent_history_record(record.run_id, agent_id, row, index, context))
    else:
        with record.trace_path.open("r", encoding="utf-8") as handle:
            for index, raw_line in enumerate(handle):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_agent_id = str(row.get("agent_id") or FOCAL_AGENT_ID)
                if row_agent_id != agent_id:
                    continue
                records.append(_agent_history_record(record.run_id, agent_id, row, index, context))
    records.sort(key=lambda item: (item["step"], item["timestamp"]))
    return {
        "run_id": record.run_id,
        "agent_id": agent_id,
        "trace_sha256": record.trace_sha256,
        "total_records": len(records),
        "records": records,
    }
