from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from heimdall_run_view.adapter import RunCatalog, RunRecord, build_precomputed_run
from heimdall_run_view.database import DatabaseUnavailableError
from heimdall_run_view.service import app


def _write_run(root: Path) -> tuple[str, RunRecord]:
    run_id = "db-parity"
    run_dir = root / "research" / "llm" / "ai-society" / "runs" / run_id
    run_dir.mkdir(parents=True)
    rows = [
        {
            "run_id": run_id,
            "step": 0,
            "timestamp": "2026-04-02T12:00:00Z",
            "agent_id": "agent-000",
            "archetype": "p2h",
            "llm_id": "L5",
            "forecaster_id": "F8",
            "decision": {
                "action": "bid",
                "side": "up",
                "quantity_mwh": 2.0,
                "limit_price_eur_mwh": 67.13,
            },
            "verifier_accepted": True,
            "market_price_eur_mwh": 80.0,
            "forecast_interval_eur_mwh": [70.0, 90.0],
        },
        {
            "run_id": run_id,
            "step": 1,
            "timestamp": "2026-04-02T12:15:00Z",
            "agent_id": "agent-000",
            "archetype": "p2h",
            "decision": {"action": "watch"},
            "market_price_eur_mwh": 81.0,
        },
    ]
    trace_path = run_dir / "traces.jsonl"
    trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    disk_record = RunCatalog(root).get(run_id)
    db_record = RunRecord(
        run_id=run_id,
        trace_path=trace_path,
        manifest_path=None,
        trace_sha256=disk_record.trace_sha256,
        total_steps=disk_record.total_steps,
        trace_rows=rows,
        summary_json={},
        bid_evaluation_rows=[],
        source_run_dir=str(run_dir),
    )
    return run_id, db_record


class FakeStore:
    def __init__(self, record: RunRecord) -> None:
        self.record = record
        self.specs: dict[str, dict[str, Any]] = {}
        self.templates: list[dict[str, Any]] = []

    def list_runs(self) -> list[dict[str, Any]]:
        return [
            {
                "run_id": self.record.run_id,
                "total_steps": self.record.total_steps,
                "trace_sha256": self.record.trace_sha256,
            }
        ]

    def get(self, run_id: str) -> RunRecord:
        if run_id != self.record.run_id:
            raise KeyError(run_id)
        return self.record

    def ensure_schema(self) -> None:
        return None

    def list_agent_templates(self) -> list[dict[str, Any]]:
        return self.templates

    def save_agent_template(self, template: dict[str, Any]) -> dict[str, Any]:
        self.templates.append(template)
        return template

    def delete_agent_template(self, template_id: str) -> None:
        before = len(self.templates)
        self.templates = [t for t in self.templates if t.get("template_id") != template_id]
        if len(self.templates) == before:
            raise KeyError(template_id)

    def save_society_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        self.specs[str(spec["society_id"])] = spec
        return spec

    def get_society_spec(self, society_id: str) -> dict[str, Any]:
        return self.specs[society_id]


class BrokenStore:
    def list_runs(self) -> list[dict[str, Any]]:
        raise DatabaseUnavailableError("offline")

    def get(self, run_id: str) -> RunRecord:
        raise DatabaseUnavailableError("offline")


def test_database_backed_precomputed_matches_disk(tmp_path: Path, monkeypatch) -> None:
    run_id, db_record = _write_run(tmp_path)
    disk_payload = build_precomputed_run(RunCatalog(tmp_path).get(run_id))
    monkeypatch.setattr("heimdall_run_view.service._database_store", lambda: FakeStore(db_record))

    response = TestClient(app).get(f"/v1/runs/{run_id}/precomputed")

    assert response.status_code == 200
    assert response.json() == disk_payload


def test_database_unavailable_falls_back_to_disk(tmp_path: Path, monkeypatch) -> None:
    run_id, _db_record = _write_run(tmp_path)
    monkeypatch.setenv("HEIMDALL_RUN_VIEW_ROOT", str(tmp_path))
    broken_store = BrokenStore()
    monkeypatch.setattr("heimdall_run_view.service._database_store", lambda: broken_store)

    response = TestClient(app).get(f"/v1/runs/{run_id}/precomputed")

    assert response.status_code == 200
    assert response.json()["run_id"] == run_id


def test_agent_template_delete_round_trips_through_store(tmp_path: Path, monkeypatch) -> None:
    _run_id, db_record = _write_run(tmp_path)
    store = FakeStore(db_record)
    monkeypatch.setattr("heimdall_run_view.service._database_store", lambda: store)
    client = TestClient(app)

    template = {
        "template_id": "aggressive-p2h",
        "label": "Aggressive P2H",
        "category": "action",
        "archetype": "p2h",
    }
    assert client.post("/v1/agent-templates", json=template).status_code == 200
    assert any(
        t["template_id"] == "aggressive-p2h"
        for t in client.get("/v1/agent-templates").json()["templates"]
    )

    delete_response = client.delete("/v1/agent-templates/aggressive-p2h")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"status": "deleted", "template_id": "aggressive-p2h"}
    assert not any(
        t["template_id"] == "aggressive-p2h"
        for t in client.get("/v1/agent-templates").json()["templates"]
    )

    # Deleting again (or an unknown id) is a 404, not a silent success.
    assert client.delete("/v1/agent-templates/aggressive-p2h").status_code == 404


def test_society_spec_round_trips_through_store(tmp_path: Path, monkeypatch) -> None:
    _run_id, db_record = _write_run(tmp_path)
    store = FakeStore(db_record)
    monkeypatch.setattr("heimdall_run_view.service._database_store", lambda: store)

    spec = {
        "society_id": "custom-society-001",
        "agents": [
            {"agent_id": "agent-custom-001", "archetype": "custom", "asset": {"capacity_mw": 12}}
        ],
    }
    save_response = TestClient(app).post("/v1/society-specs", json=spec)
    get_response = TestClient(app).get("/v1/society-specs/custom-society-001")

    assert save_response.status_code == 200
    assert get_response.status_code == 200
    assert get_response.json()["society_spec"] == spec
