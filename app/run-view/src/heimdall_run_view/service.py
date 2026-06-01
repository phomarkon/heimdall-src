from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from heimdall_run_view.adapter import (
    RunCatalog,
    RunRecord,
    build_agent_history,
    build_precomputed_run,
)
from heimdall_run_view.database import (
    BUILTIN_AGENT_TEMPLATES,
    DatabaseUnavailableError,
    PostgresRunStore,
)

app = FastAPI(title="heimdall-run-view", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/runs")
def list_runs() -> dict[str, object]:
    # Hybrid mode: Postgres only owns the catalog once runs have actually been
    # ingested into it. When the runs table is empty (the common case where PG is
    # used solely for society-spec / agent-template persistence), fall back to the
    # disk catalog so an empty database never blanks the run list.
    store = _database_store()
    if store is not None:
        try:
            runs = store.list_runs()
            if runs:
                return {"runs": runs}
        except DatabaseUnavailableError:
            pass
    return {"runs": RunCatalog().list_runs()}


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str) -> dict[str, object]:
    try:
        record = _get_record(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}") from exc
    return {
        "run_id": record.run_id,
        "total_steps": record.total_steps,
        "trace_sha256": record.trace_sha256,
        "status": record.status,
        "trace_path": str(record.trace_path),
    }


@app.get("/v1/runs/{run_id}/society")
def get_snapshot(run_id: str, step: int = 0) -> dict[str, object]:
    try:
        record = _get_record(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}") from exc
    run = build_precomputed_run(record)
    safe_step = max(0, min(int(step), run["total_steps"] - 1))
    return run["snapshots"][safe_step]


@app.get("/v1/runs/{run_id}/precomputed")
def get_precomputed(run_id: str) -> dict[str, object]:
    try:
        record = _get_record(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}") from exc
    return build_precomputed_run(record)


@app.get("/v1/runs/{run_id}/agents/{agent_id}/history")
def get_agent_history(run_id: str, agent_id: str) -> dict[str, object]:
    try:
        record = _get_record(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown run_id {run_id}") from exc
    return build_agent_history(record, agent_id)


@app.get("/v1/agent-templates")
def list_agent_templates() -> dict[str, object]:
    store = _database_store()
    if store is None:
        return {"templates": BUILTIN_AGENT_TEMPLATES, "database": "unavailable"}
    try:
        return {"templates": store.list_agent_templates(), "database": "available"}
    except DatabaseUnavailableError:
        return {"templates": BUILTIN_AGENT_TEMPLATES, "database": "unavailable"}


@app.post("/v1/agent-templates")
def save_agent_template(template: dict[str, Any]) -> dict[str, object]:
    store = _require_database_store()
    try:
        return {"template": store.save_agent_template(template)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/v1/agent-templates/{template_id}")
def delete_agent_template(template_id: str) -> dict[str, object]:
    store = _require_database_store()
    try:
        store.delete_agent_template(template_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown template_id {template_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "deleted", "template_id": template_id}


@app.post("/v1/society-specs")
def save_society_spec(spec: dict[str, Any]) -> dict[str, object]:
    store = _require_database_store()
    try:
        return {"society_spec": store.save_society_spec(spec)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/v1/society-specs/{society_id}")
def get_society_spec(society_id: str) -> dict[str, object]:
    store = _require_database_store()
    try:
        return {"society_spec": store.get_society_spec(society_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown society_id {society_id}") from exc


def _get_record(run_id: str) -> RunRecord:
    store = _database_store()
    if store is not None:
        try:
            return store.get(run_id)
        except DatabaseUnavailableError:
            pass
        except KeyError:
            pass
    return RunCatalog().get(run_id)


def _database_store() -> PostgresRunStore | None:
    try:
        return PostgresRunStore()
    except DatabaseUnavailableError:
        return None


def _require_database_store() -> PostgresRunStore:
    store = _database_store()
    if store is None:
        raise HTTPException(status_code=503, detail="Postgres run store is not configured")
    try:
        store.ensure_schema()
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Postgres run store is unavailable") from exc
    return store
