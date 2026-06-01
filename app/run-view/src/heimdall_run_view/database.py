"""Postgres-backed run store (facade over focused repository classes)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from heimdall_run_view._db import (
    BUILTIN_AGENT_TEMPLATES,
    AgentTemplateRepository,
    DatabaseUnavailableError,
    SocietySpecRepository,
    _BaseRepository,
)
from heimdall_run_view._utils import (
    RunRecord,
    _file_sha256,
    _iso_z,
    _optional_float,
    _optional_int,
    _parse_timestamp,
    _read_bid_evaluations,
    _read_json,
    _read_jsonl,
    _read_run_summary,
    _scan_trace_metadata,
)

__all__ = ["BUILTIN_AGENT_TEMPLATES", "DatabaseUnavailableError", "PostgresRunStore"]


class PostgresRunStore:
    """Composes run, template, and society-spec repositories behind a single API."""

    def __init__(self, database_url: str | None = None) -> None:
        url = database_url or os.getenv("HEIMDALL_RUN_VIEW_DATABASE_URL")
        if not url:
            raise DatabaseUnavailableError("HEIMDALL_RUN_VIEW_DATABASE_URL is not set")
        self._base = _BaseRepository(url)
        self._templates = AgentTemplateRepository(url)
        self._society = SocietySpecRepository(url)

    # -- runs ----------------------------------------------------------------

    def list_runs(self) -> list[dict[str, Any]]:
        with self._base._connect() as conn:
            rows = conn.execute(
                """
                select run_id, status, started_at, total_steps, summary_json, trace_sha256, source_run_dir
                from runs
                order by run_id
                """
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "total_steps": row["total_steps"],
                "trace_sha256": row["trace_sha256"],
                "status": row["status"],
                "trace_path": str(Path(row["source_run_dir"] or "") / "traces.jsonl"),
                "setup_id": _setup_id_from_path(row["source_run_dir"], row["run_id"]),
                "setup_label": _label(_setup_id_from_path(row["source_run_dir"], row["run_id"])),
                "window_label": row["run_id"],
                "start_timestamp": _iso_z(row["started_at"]) if row["started_at"] else None,
                "has_evaluation": bool(row["summary_json"]),
                "pnl_eur": _optional_float((row["summary_json"] or {}).get("cumulative_pnl_eur")),
                "bid_action_count": _optional_int(
                    (row["summary_json"] or {}).get("bid_action_count")
                ),
                "cleared_mwh": _optional_float((row["summary_json"] or {}).get("cleared_mwh")),
                "forecaster_id": None,
                "control_mode": None,
            }
            for row in rows
        ]

    def get(self, run_id: str) -> RunRecord:
        with self._base._connect() as conn:
            run = conn.execute(
                """
                select run_id, status, started_at, total_steps, config_json, summary_json, trace_sha256, source_run_dir
                from runs
                where run_id = %s
                """,
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(run_id)
            traces = conn.execute(
                """
                select event_json
                from trace_events
                where run_id = %s
                order by event_index
                """,
                (run_id,),
            ).fetchall()
            evaluation = conn.execute(
                "select summary_json, bid_evaluations_json from evaluations where run_id = %s",
                (run_id,),
            ).fetchone()
        source_dir = Path(run["source_run_dir"] or ".")
        return RunRecord(
            run_id=run["run_id"],
            trace_path=source_dir / "traces.jsonl",
            manifest_path=source_dir / "manifest.json",
            trace_sha256=run["trace_sha256"],
            total_steps=run["total_steps"],
            status=run["status"],
            trace_rows=[row["event_json"] for row in traces],
            summary_json=(evaluation or {}).get("summary_json") if evaluation else None,
            bid_evaluation_rows=(evaluation or {}).get("bid_evaluations_json")
            if evaluation
            else None,
            config_json=run["config_json"],
            source_run_dir=run["source_run_dir"],
        )

    def ingest_run(self, run_dir: Path, *, repo_root: Path | None = None) -> str:
        repo_root = repo_root or _repo_root()
        trace_path = run_dir / "traces.jsonl"
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)
        metadata = _scan_trace_metadata(trace_path)
        if metadata is None:
            raise ValueError(f"no trace rows found in {trace_path}")
        first, total_steps = metadata
        run_id = str(first.get("run_id") or run_dir.name)
        trace_sha = _file_sha256(trace_path)
        config_json = _load_config_json(run_dir)
        disk_record = RunRecord(
            run_id=run_id,
            trace_path=trace_path,
            manifest_path=_manifest_path(repo_root, run_id, run_dir),
            trace_sha256=trace_sha,
            total_steps=total_steps,
        )
        manifest = _read_json(disk_record.manifest_path) if disk_record.manifest_path else {}
        run_summary = _read_json(run_dir / "summary.json")
        evaluation_summary = _read_run_summary(disk_record, manifest)
        bid_rows = _read_bid_evaluations(disk_record, manifest)
        traces = _read_jsonl(trace_path)
        started_at = first.get("timestamp") or first.get("utc_timestamp")
        with self._base._connect() as conn:
            self._base.ensure_schema(conn)
            with conn.transaction():
                conn.execute(
                    """
                    insert into runs (run_id, status, started_at, total_steps, config_json, summary_json, trace_sha256, source_run_dir)
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (run_id) do update set
                        status = excluded.status,
                        started_at = excluded.started_at,
                        total_steps = excluded.total_steps,
                        config_json = excluded.config_json,
                        summary_json = excluded.summary_json,
                        trace_sha256 = excluded.trace_sha256,
                        source_run_dir = excluded.source_run_dir
                    """,
                    (
                        run_id,
                        "completed",
                        _parse_timestamp(started_at),
                        total_steps,
                        self._base._jsonb(config_json),
                        self._base._jsonb(evaluation_summary or run_summary),
                        trace_sha,
                        str(run_dir),
                    ),
                )
                conn.execute("delete from agents where run_id = %s", (run_id,))
                conn.execute("delete from trace_events where run_id = %s", (run_id,))
                conn.execute("delete from evaluations where run_id = %s", (run_id,))
                conn.executemany(
                    """
                    insert into trace_events (run_id, event_index, step, timestamp, agent_id, event_json)
                    values (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            run_id,
                            index,
                            int(row.get("step", index)),
                            _parse_timestamp(row.get("timestamp") or row.get("utc_timestamp")),
                            str(row.get("agent_id") or "agent-000"),
                            self._base._jsonb(row),
                        )
                        for index, row in enumerate(traces)
                    ],
                )
                conn.executemany(
                    """
                    insert into agents (run_id, agent_id, archetype, display_name, persona_json, asset_json, is_custom)
                    values (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            run_id,
                            agent_id,
                            str(
                                row.get("archetype")
                                or (row.get("persona") or {}).get("archetype")
                                or "custom"
                            ),
                            str((row.get("persona") or {}).get("display_name") or agent_id),
                            self._base._jsonb(
                                row.get("persona") or _persona_from_row(row, agent_id)
                            ),
                            self._base._jsonb(row.get("asset") or {}),
                            bool(
                                row.get("is_custom")
                                or str(row.get("archetype") or "").startswith("custom")
                            ),
                        )
                        for agent_id, row in _agent_rows(traces).items()
                    ],
                )
                conn.execute(
                    """
                    insert into evaluations (run_id, summary_json, bid_evaluations_json, created_at)
                    values (%s, %s, %s, now())
                    on conflict (run_id) do update set
                        summary_json = excluded.summary_json,
                        bid_evaluations_json = excluded.bid_evaluations_json,
                        created_at = excluded.created_at
                    """,
                    (run_id, self._base._jsonb(evaluation_summary), self._base._jsonb(bid_rows)),
                )
        return run_id

    # -- agent templates (delegated) -----------------------------------------

    def list_agent_templates(self) -> list[dict[str, Any]]:
        return self._templates.list()

    def save_agent_template(self, template: dict[str, Any]) -> dict[str, Any]:
        return self._templates.save(template)

    def delete_agent_template(self, template_id: str) -> None:
        return self._templates.delete(template_id)

    # -- society specs (delegated) -------------------------------------------

    def save_society_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self._society.save(spec)

    def get_society_spec(self, society_id: str) -> dict[str, Any]:
        return self._society.get(society_id)

    # -- schema management ---------------------------------------------------

    def ensure_schema(self, conn: Any | None = None) -> None:
        self._base.ensure_schema(conn)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_config_json(run_dir: Path) -> dict[str, Any]:
    for name in ("config.json", "run_config.json", "society_spec.json"):
        path = run_dir / name
        if path.exists():
            return _read_json(path)
    for name in ("config.yaml", "config.yml"):
        path = run_dir / name
        if path.exists():
            try:
                return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:
                return {"source_config_path": str(path)}
    return {}


def _manifest_path(repo_root: Path, run_id: str, run_dir: Path) -> Path | None:
    candidates = [repo_root / "research" / "llm" / "evaluations" / run_id / "manifest.json", run_dir / "manifest.json"]
    return next((path for path in candidates if path.exists()), None)


def _agent_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    agents: dict[str, dict[str, Any]] = {}
    for row in rows:
        agent_id = str(row.get("agent_id") or "agent-000")
        agents.setdefault(agent_id, row)
    return agents


def _persona_from_row(row: dict[str, Any], agent_id: str) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "archetype": row.get("archetype"),
        "forecaster_id": row.get("forecaster_id"),
        "llm_id": row.get("llm_id"),
    }


def _repo_root() -> Path:
    return Path(os.getenv("HEIMDALL_RUN_VIEW_ROOT") or Path(__file__).resolve().parents[4])


def _setup_id_from_path(source_run_dir: str | None, run_id: str) -> str:
    if not source_run_dir:
        return run_id.split("-", maxsplit=1)[0] if "-" in run_id else "standalone"
    parent = Path(source_run_dir).parent.name
    return parent if parent and parent != "runs" else run_id.split("-", maxsplit=1)[0]


def _label(value: str) -> str:
    return value.replace("-", " ").title() if value else "Standalone"
