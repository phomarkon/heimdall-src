"""Contract / integration test for the run-view <-> frontend seam.

The frontend (apps/frontend) consumes these payloads and parses them against the
TypeScript interfaces in `src/types/heimdall.ts`. The unit/component tests there mock
the network, and the e2e suite runs in mock mode, so nothing else verifies that the
*real* backend output matches what the frontend requires. This test does: it builds a
realistic multi-agent run (bids + realized outcomes + broadcast comm + a baseline
evaluation) and asserts every endpoint emits the exact fields and types the frontend
relies on — including the ones that previously drifted (`edges`, `focal_baselines`,
`priority_signal.grounding`).

When run with HEIMDALL_WRITE_CONTRACT_FIXTURES=1 it also writes the payloads to
apps/frontend/src/test-fixtures/ so the frontend integration test renders the real
backend shape.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient
from heimdall_run_view.service import app

RUN_ID = "contract-smoke"
AGENTS = [
    ("agent-000", "p2h"),
    ("agent-001", "ev"),
    ("agent-002", "wind"),
]


def _comm_call() -> dict[str, Any]:
    return {
        "name": "society_communication_context",
        "ok": True,
        "result": {
            "context": {"roster": [{"agent_id": a, "archetype": arch} for a, arch in AGENTS]}
        },
    }


def _sim_call(side: str, accepted: bool) -> dict[str, Any]:
    return {
        "name": "simulate_bid",
        "arguments": {"side": side, "quantity_mwh": 2.0, "limit_price_eur_mwh": 60.0},
        "ok": True,
        "result": {
            "accepted": accepted,
            "worst_case_profit_eur": 5.0,
            "rough_expected_profit_eur": 9.0,
            "reason_codes": [] if accepted else ["price_not_crossed"],
        },
    }


def _trace_row(
    step: int, agent_id: str, archetype: str, action: str, side: str | None
) -> dict[str, Any]:
    decision = {
        "action": action,
        "side": side,
        "quantity_mwh": 2.0 if action == "bid" else None,
        "limit_price_eur_mwh": 60.0 if action == "bid" else None,
        "rationale": "contract row",
    }
    calls = [_comm_call()]
    if action == "bid":
        calls.append(_sim_call(side or "up", accepted=True))
    return {
        "run_id": RUN_ID,
        "step": step,
        "timestamp": f"2026-04-02T{12 + step:02d}:00:00Z",
        "agent_id": agent_id,
        "agent_role": "society_chair" if agent_id == "agent-002" else "action_agent",
        "archetype": archetype,
        "llm_id": "L5",
        "forecaster_id": "F8",
        "market_price_eur_mwh": 80.0,
        "forecast_interval_eur_mwh": [70.0, 90.0],
        "decision": decision,
        "rationale": "contract row",
        "verifier_accepted": action == "bid",
        "verifier_reason_codes": [],
        "tool_calls": calls,
    }


def _write_contract_run(root: Path) -> None:
    run_dir = root / "research" / "llm" / "ai-society" / "runs" / RUN_ID
    eval_dir = root / "research" / "llm" / "evaluations" / RUN_ID
    run_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    for step in range(2):
        # two agents bid the same side (-> consensus edge), all consume comm (-> broadcast)
        rows.append(_trace_row(step, "agent-000", "p2h", "bid", "up"))
        rows.append(_trace_row(step, "agent-001", "ev", "bid", "up"))
        rows.append(_trace_row(step, "agent-002", "wind", "watch", None))
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (eval_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0.0", "run_id": RUN_ID}), encoding="utf-8"
    )

    bid_rows = [
        {
            "step": 0,
            "agent_id": "agent-000",
            "status": "filled",
            "cleared_mwh": 2.0,
            "realized_profit_eur": 120.0,
            "market_price_eur_mwh": 80.0,
            "verifier_accepted": True,
        },
        {
            "step": 0,
            "agent_id": "agent-001",
            "status": "price_not_crossed",
            "cleared_mwh": 0.0,
            "realized_profit_eur": 0.0,
            "market_price_eur_mwh": 80.0,
            "verifier_accepted": True,
        },
        {
            "step": 1,
            "agent_id": "agent-000",
            "status": "filled",
            "cleared_mwh": 2.0,
            "realized_profit_eur": 60.0,
            "market_price_eur_mwh": 80.0,
            "verifier_accepted": True,
        },
    ]
    pd.DataFrame(bid_rows).to_parquet(eval_dir / "bid_evaluations.parquet")

    baseline = root / "research" / "llm" / "evaluations" / "mi00-baseline-profitguard-24-q32"
    baseline.mkdir(parents=True)
    (baseline / "run_summary.json").write_text(
        json.dumps(
            {
                "run_id": "mi00-baseline-profitguard-24-q32",
                "cumulative_pnl_eur": 0.0,
                "realized_profit_eur": 0.0,
                "downside_cvar_95_eur": 0.0,
                "fill_rate": 0.0,
                "bid_count": 24,
                "regret_eur": 6175.18,
            }
        ),
        encoding="utf-8",
    )


# --- contract assertions (mirror src/types/heimdall.ts) ---


def _require(obj: dict[str, Any], field: str, types: type | tuple[type, ...], where: str) -> Any:
    assert field in obj, f"{where}: missing required field '{field}'"
    value = obj[field]
    assert isinstance(value, types), f"{where}.{field}: expected {types}, got {type(value)}"
    return value


def _assert_persona(p: dict[str, Any], where: str) -> None:
    for f in (
        "agent_id",
        "display_name",
        "archetype",
        "risk_attitude",
        "sophistication",
        "llm_family",
        "forecaster",
    ):
        _require(p, f, str, where)
    _require(p, "info_latency_min", (int, float), where)
    _require(p, "capacity_mw", (int, float), where)
    assert "storage_mwh" in p


def _assert_priority_signal(s: dict[str, Any], where: str) -> None:
    _require(s, "score", (int, float), where)
    _require(s, "percentile", (int, float), where)
    tier = _require(s, "tier", str, where)
    assert tier in {"low", "watch", "medium", "high", "critical"}, f"{where}: bad tier {tier}"
    _require(s, "label", str, where)
    _require(s, "drivers", list, where)
    _require(s, "risks", list, where)
    if "grounding" in s:
        assert s["grounding"] in {"realized_outcome", "forward_estimate"}


def _assert_trace(t: dict[str, Any], where: str) -> None:
    for f in ("run_id", "timestamp", "agent_id", "reasoning"):
        _require(t, f, str, where)
    _require(t, "step", int, where)
    _assert_persona(_require(t, "persona", dict, where), f"{where}.persona")
    _require(t, "tool_calls", list, where)
    action = _require(t, "proposed_action", dict, where)
    for f in ("market", "direction", "delivery_quarter"):
        _require(action, f, str, f"{where}.proposed_action")
    verdict = _require(t, "verifier_verdict", dict, where)
    _require(verdict, "accepted", bool, f"{where}.verifier_verdict")
    ci = _require(verdict, "conformal_interval", dict, f"{where}.verifier_verdict")
    for f in ("horizon_minutes", "quantile_low", "quantile_high", "alpha"):
        _require(ci, f, (int, float), f"{where}.conformal_interval")


def _maybe_write_fixtures(precomputed: dict[str, Any], history: dict[str, Any]) -> None:
    if os.getenv("HEIMDALL_WRITE_CONTRACT_FIXTURES") != "1":
        return
    out = Path(__file__).resolve().parents[3] / "apps" / "frontend" / "src" / "test-fixtures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "precomputed.contract.json").write_text(
        json.dumps(precomputed, indent=2), encoding="utf-8"
    )
    (out / "agent-history.contract.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


def test_frontend_contract_precomputed_society_history_catalog(tmp_path: Path, monkeypatch) -> None:
    _write_contract_run(tmp_path)
    monkeypatch.setenv("HEIMDALL_RUN_VIEW_ROOT", str(tmp_path))
    client = TestClient(app)

    # --- /v1/runs catalog ---
    catalog = client.get("/v1/runs").json()
    _require(catalog, "runs", list, "catalog")
    row = next(r for r in catalog["runs"] if r["run_id"] == RUN_ID)
    for f in ("run_id", "trace_sha256", "status", "trace_path"):
        _require(row, f, str, "catalog.row")
    _require(row, "total_steps", int, "catalog.row")

    # --- /v1/runs/{id}/precomputed ---
    run = client.get(f"/v1/runs/{RUN_ID}/precomputed").json()
    _require(run, "run_id", str, "precomputed")
    _require(run, "total_steps", int, "precomputed")
    snapshots = _require(run, "snapshots", list, "precomputed")
    _require(run, "market_series", list, "precomputed")
    assert snapshots, "precomputed: no snapshots"

    baselines = _require(run, "focal_baselines", list, "precomputed")
    assert baselines, "precomputed.focal_baselines empty"
    for b in baselines:
        for f in ("run_id", "label", "kind", "status", "source"):
            _require(b, f, str, "focal_baseline")
        _require(b, "n_runs", int, "focal_baseline")
        assert b["kind"] in {"baseline", "ablation"}

    snap = snapshots[0]
    for f in ("run_id",):
        _require(snap, f, str, "snapshot")
    _require(snap, "step", int, "snapshot")
    nodes = _require(snap, "nodes", list, "snapshot")
    edges = _require(snap, "edges", list, "snapshot")
    _assert_trace(_require(snap, "selected_trace", dict, "snapshot"), "snapshot.selected_trace")
    market = _require(snap, "market", dict, "snapshot")
    _require(market, "timestamp", str, "market")
    for f in (
        "dk1_price_eur_per_mwh",
        "dk2_price_eur_per_mwh",
        "mfrr_price_eur_per_mwh",
        "gate_closure_minutes",
    ):
        _require(market, f, (int, float), "market")
    _require(market, "events", list, "market")
    _assert_priority_signal(
        _require(market, "priority_signal", dict, "market"), "market.priority_signal"
    )
    health = _require(snap, "health", dict, "snapshot")
    for f in ("coverage", "verifier_acceptance_rate", "cumulative_pnl_eur"):
        _require(health, f, (int, float), "health")

    # nodes
    focal_count = 0
    for node in nodes:
        _require(node, "id", str, "node")
        _assert_persona(_require(node, "persona", dict, "node"), "node.persona")
        for f in ("x", "y", "open_position_mw", "pnl_eur"):
            _require(node, f, (int, float), "node")
        _require(node, "belief", str, "node")
        focal_count += int(_require(node, "is_focal", bool, "node"))
    assert focal_count == 1, "exactly one focal node expected"

    # edges — the seam that used to be empty; here we must see consensus + broadcast
    assert edges, "snapshot.edges empty — frontend graph would show no connections"
    kinds = set()
    for edge in edges:
        for f in ("id", "source", "target", "kind", "market", "label", "detail"):
            _require(edge, f, str, "edge")
        _require(edge, "strength", (int, float), "edge")
        _require(edge, "started_step", int, "edge")
        assert edge["kind"] in {"consensus", "broadcast"}, f"bad edge kind {edge['kind']}"
        kinds.add(edge["kind"])
    assert {"consensus", "broadcast"} <= kinds, f"expected both edge kinds, got {kinds}"

    # outcome-grounded priority on a run with realized profit
    assert market["priority_signal"].get("grounding") == "realized_outcome"

    # --- /v1/runs/{id}/agents/{agent}/history ---
    history = client.get(f"/v1/runs/{RUN_ID}/agents/agent-000/history").json()
    for f in ("run_id", "agent_id", "trace_sha256"):
        _require(history, f, str, "history")
    _require(history, "total_records", int, "history")
    records = _require(history, "records", list, "history")
    assert records, "history: no records"
    rec = records[0]
    for f in ("run_id", "timestamp", "agent_id", "rationale"):
        _require(rec, f, str, "history.record")
    _require(rec, "step", int, "history.record")
    _require(rec, "decision", dict, "history.record")
    _require(rec, "tool_calls", list, "history.record")
    verifier = _require(rec, "verifier", dict, "history.record")
    _require(verifier, "reason_codes", list, "history.record.verifier")
    assert "accepted" in verifier and "stage_failed" in verifier

    _maybe_write_fixtures(run, history)
