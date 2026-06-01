from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from heimdall_run_view.adapter import (
    RunCatalog,
    build_agent_history,
    build_precomputed_run,
)
from heimdall_run_view.service import app


def _write_run(root: Path) -> str:
    run_id = "adapter-smoke"
    run_dir = root / "research" / "llm" / "ai-society" / "runs" / run_id
    eval_dir = root / "research" / "llm" / "evaluations" / run_id
    run_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
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
            "verifier_reason_codes": [],
            "market_price_eur_mwh": 80.0,
            "forecast_interval_eur_mwh": [70.0, 90.0],
            "rationale": "accepted trace",
            "tool_calls": [
                {
                    "name": "simulate_bid",
                    "ok": True,
                    "result": {
                        "accepted": True,
                        "worst_case_profit_eur": 12.0,
                        "expected_spread_eur_mwh": 8.0,
                        "signals": {"up_edge_lower_minus_spot_eur_mwh": 3.5},
                    },
                }
            ],
        },
        {
            "run_id": run_id,
            "step": 1,
            "timestamp": "2026-04-02T12:15:00Z",
            "agent_id": "agent-000",
            "archetype": "p2h",
            "llm_id": "L5",
            "forecaster_id": "F8",
            "decision": {"action": "abstain"},
            "verifier_accepted": None,
            "verifier_reason_codes": [],
            "market_price_eur_mwh": 81.0,
            "forecast_interval_eur_mwh": [70.0, 90.0],
        },
        {
            "run_id": run_id,
            "step": 2,
            "timestamp": "2026-04-02T12:30:00Z",
            "agent_id": "agent-000",
            "archetype": "p2h",
            "llm_id": "L5",
            "forecaster_id": "F8",
            "decision": {
                "action": "bid",
                "side": "down",
                "quantity_mwh": 3.0,
                "limit_price_eur_mwh": 60.0,
            },
            "verifier_accepted": False,
            "verifier_reason_codes": ["gate_closed"],
            "market_price_eur_mwh": 55.0,
            "forecast_interval_eur_mwh": [50.0, 65.0],
        },
    ]
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (eval_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "1.0.0", "run_id": run_id}), encoding="utf-8"
    )
    (eval_dir / "run_summary.json").write_text(
        json.dumps({"cumulative_pnl_eur": 123.45, "bid_action_count": 2, "cleared_mwh": 4.5}),
        encoding="utf-8",
    )
    return run_id


def test_catalog_lists_run_selector_metadata(tmp_path: Path) -> None:
    run_id = _write_run(tmp_path)

    runs = RunCatalog(tmp_path).list_runs()
    row = next(item for item in runs if item["run_id"] == run_id)

    assert row["setup_id"] == "adapter"
    assert row["setup_label"] == "Adapter"
    assert row["window_label"] == run_id
    assert row["start_timestamp"] == "2026-04-02T12:00:00Z"
    assert row["has_evaluation"] is True
    assert row["pnl_eur"] == 123.45
    assert row["bid_action_count"] == 2
    assert row["cleared_mwh"] == 4.5
    assert row["forecaster_id"] == "f8"
    assert row["control_mode"] is None


def test_catalog_lists_trace_only_runs_without_evaluation(tmp_path: Path) -> None:
    run_id = "trace-only-apr02-96-proxy-controls"
    run_dir = tmp_path / "research" / "llm" / "ai-society" / "runs" / "mixed20-full-days" / run_id
    run_dir.mkdir(parents=True)
    rows = [
        {"run_id": run_id, "step": 0, "timestamp": "2026-04-02T00:00:00Z", "forecaster_id": "F8"},
        {"run_id": run_id, "step": 95, "timestamp": "2026-04-02T23:45:00Z", "forecaster_id": "F8"},
    ]
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    row = RunCatalog(tmp_path).list_runs()[0]

    assert row["run_id"] == run_id
    assert row["total_steps"] == 96
    assert row["setup_label"] == "Mixed-20 full days"
    assert row["window_label"] == "Apr 02 / 96 ticks / proxy controls"
    assert row["has_evaluation"] is False
    assert row["pnl_eur"] is None
    assert row["bid_action_count"] is None
    assert row["cleared_mwh"] is None
    assert row["control_mode"] == "proxy controls"


def test_catalog_counts_unique_steps_for_multi_agent_traces(tmp_path: Path) -> None:
    run_id = "mixed20-apr02-96-real-controls"
    run_dir = tmp_path / "research" / "llm" / "ai-society" / "runs" / "mixed20-full-days" / run_id
    run_dir.mkdir(parents=True)
    rows = [
        {
            "run_id": run_id,
            "step": step,
            "timestamp": f"2026-04-02T{step // 4:02d}:{(step % 4) * 15:02d}:00Z",
            "agent_id": f"agent-{agent:03d}",
            "forecaster_id": "F8",
        }
        for step in range(96)
        for agent in range(20)
    ]
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    row = RunCatalog(tmp_path).list_runs()[0]

    assert row["run_id"] == run_id
    assert row["total_steps"] == 96


def test_catalog_builds_frontend_snapshots(tmp_path: Path) -> None:
    run_id = _write_run(tmp_path)
    record = RunCatalog(tmp_path).get(run_id)

    run = build_precomputed_run(record)

    assert run["run_id"] == run_id
    assert run["total_steps"] == 3
    assert run["forecaster_leaderboard"]
    assert run["forecaster_summary"]["active_forecaster_id"] == "f8"
    assert any(event["kind"] == "accepted_bid" for event in run["snapshots"][0]["market"]["events"])
    assert run["snapshots"][0]["selected_trace"]["verifier_verdict"]["accepted"] is True
    assert run["snapshots"][0]["selected_trace"]["realized_outcome"]["fill_mw"] == 2.0
    assert run["snapshots"][0]["forecast_diagnostics"]["covered"] is True
    assert run["snapshots"][0]["forecast_diagnostics"]["interval_width_eur_mwh"] == 20.0
    assert run["snapshots"][0]["forecast_diagnostics"]["up_edge_eur_mwh"] == 3.5
    assert any(event["kind"] == "rejected_bid" for event in run["snapshots"][2]["market"]["events"])
    assert run["snapshots"][2]["selected_trace"]["verifier_verdict"]["stage_failed"] == "physical"


def test_society_edges_derive_consensus_and_broadcast(tmp_path: Path) -> None:
    run_id = "edges-smoke"
    run_dir = tmp_path / "research" / "llm" / "ai-society" / "runs" / run_id
    run_dir.mkdir(parents=True)
    comm = {
        "name": "society_communication_context",
        "ok": True,
        "result": {"context": {"roster": []}},
    }
    rows = [
        {
            "run_id": run_id,
            "step": 0,
            "timestamp": "2026-04-02T12:00:00Z",
            "agent_id": "agent-000",
            "archetype": "p2h",
            "agent_role": "society_chair",
            "decision": {"action": "bid", "side": "up", "quantity_mwh": 5.0},
            "tool_calls": [comm],
        },
        {
            "run_id": run_id,
            "step": 0,
            "timestamp": "2026-04-02T12:00:00Z",
            "agent_id": "agent-001",
            "archetype": "wind",
            "decision": {"action": "bid", "side": "up", "quantity_mwh": 2.0},
            "tool_calls": [comm],
        },
        {
            "run_id": run_id,
            "step": 0,
            "timestamp": "2026-04-02T12:00:00Z",
            "agent_id": "agent-002",
            "archetype": "generator",
            "decision": {"action": "bid", "side": "down", "quantity_mwh": 1.0},
            "tool_calls": [],
        },
    ]
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    run = build_precomputed_run(RunCatalog(tmp_path).get(run_id))
    edges = run["snapshots"][0]["edges"]
    kinds = {edge["kind"] for edge in edges}

    assert "consensus" in kinds  # agent-000 + agent-001 both bid up
    assert "broadcast" in kinds  # both consumed the society broadcast
    consensus = next(edge for edge in edges if edge["kind"] == "consensus")
    assert consensus["side"] == "up"
    assert consensus["source"] == "agent-000"  # p2h anchor
    # the lone "down" bid has no same-side peer, so no consensus edge for it
    assert all(not (edge["kind"] == "consensus" and edge["side"] == "down") for edge in edges)


def test_focal_baselines_read_only_existing_evaluations(tmp_path: Path, monkeypatch) -> None:
    run_id = "baseline-host"
    run_dir = tmp_path / "research" / "llm" / "ai-society" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "traces.jsonl").write_text(
        json.dumps({"run_id": run_id, "step": 0}) + "\n", encoding="utf-8"
    )
    baseline_dir = tmp_path / "research" / "llm" / "evaluations" / "mi00-baseline-profitguard-24-q32"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "run_summary.json").write_text(
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
    monkeypatch.setenv("HEIMDALL_RUN_VIEW_ROOT", str(tmp_path))

    run = build_precomputed_run(RunCatalog(tmp_path).get(run_id))
    baselines = run["focal_baselines"]

    assert len(baselines) == 1  # only the one evaluated baseline on disk, nothing fabricated
    row = baselines[0]
    assert row["label"].startswith("Profit-guard baseline")
    assert row["bid_count"] == 24
    assert row["regret_eur"] == 6175.18
    assert row["status"] == "evaluated"


def test_sparse_trace_rows_have_stable_defaults(tmp_path: Path) -> None:
    run_id = "sparse"
    run_dir = tmp_path / "research" / "llm" / "ai-society" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "traces.jsonl").write_text(
        json.dumps({"run_id": run_id, "step": 0}) + "\n", encoding="utf-8"
    )

    run = build_precomputed_run(RunCatalog(tmp_path).get(run_id))

    snapshot = run["snapshots"][0]
    assert snapshot["nodes"]
    assert snapshot["selected_trace"]["tool_calls"]
    assert snapshot["market"]["timestamp"].endswith("Z")
    assert snapshot["forecast_diagnostics"]["covered"] is None


def test_agent_history_filters_rows_and_preserves_raw_tool_calls(tmp_path: Path) -> None:
    run_id = "history-smoke"
    run_dir = tmp_path / "research" / "llm" / "ai-society" / "runs" / run_id
    run_dir.mkdir(parents=True)
    rows = [
        {
            "run_id": run_id,
            "step": 0,
            "timestamp": "2026-04-02T12:00:00Z",
            "observed_at": "2026-04-02T11:45:00Z",
            "agent_id": "agent-001",
            "zone": "DK1",
            "archetype": "p2h",
            "market_price_eur_mwh": 80.0,
            "forecast_interval_eur_mwh": [70.0, 90.0],
            "decision": {
                "action": "bid",
                "side": "up",
                "quantity_mwh": 2.0,
                "limit_price_eur_mwh": 67.13,
            },
            "verifier_accepted": True,
            "verifier_reason_codes": [],
            "rationale": "accepted with context",
            "tool_calls": [
                {
                    "name": "simulate_bid",
                    "arguments": {"side": "up", "quantity_mwh": 2.0},
                    "ok": True,
                    "result": {"accepted": True, "worst_case_profit_eur": 12.0},
                    "error": None,
                    "provenance": "llm_requested",
                }
            ],
            "tool_call_provenance_counts": {"llm_requested": 1},
        },
        {
            "run_id": run_id,
            "step": 0,
            "timestamp": "2026-04-02T12:00:00Z",
            "agent_id": "agent-002",
            "decision": {"action": "watch"},
            "tool_calls": [
                {
                    "name": "get_activation_context",
                    "arguments": {"hours": 24},
                    "ok": True,
                    "result": {"watch_score": 0.8},
                }
            ],
        },
        {
            "run_id": run_id,
            "step": 1,
            "timestamp": "2026-04-02T12:15:00Z",
            "agent_id": "agent-001",
            "decision": {"action": "watch"},
        },
    ]
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    history = build_agent_history(RunCatalog(tmp_path).get(run_id), "agent-001")

    assert history["total_records"] == 2
    assert [record["step"] for record in history["records"]] == [0, 1]
    assert history["records"][0]["decision"]["action"] == "bid"
    assert history["records"][0]["observed_at"] == "2026-04-02T11:45:00Z"
    assert history["records"][0]["tool_calls"][0]["arguments"] == {
        "side": "up",
        "quantity_mwh": 2.0,
    }
    assert history["records"][0]["tool_calls"][0]["result"]["worst_case_profit_eur"] == 12.0
    assert history["records"][0]["tool_calls"][0]["provenance"] == "llm_requested"
    assert history["records"][0]["tool_call_provenance_counts"]["llm_requested"] == 1


def test_agent_history_endpoint_uses_lazy_agent_filter(tmp_path: Path, monkeypatch) -> None:
    run_id = "history-endpoint"
    run_dir = tmp_path / "research" / "llm" / "ai-society" / "runs" / run_id
    run_dir.mkdir(parents=True)
    rows = [
        {
            "run_id": run_id,
            "step": 0,
            "timestamp": "2026-04-02T12:00:00Z",
            "agent_id": "agent-001",
            "decision": {"action": "bid"},
        },
        {
            "run_id": run_id,
            "step": 0,
            "timestamp": "2026-04-02T12:00:00Z",
            "agent_id": "agent-002",
            "decision": {"action": "watch"},
        },
    ]
    (run_dir / "traces.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("HEIMDALL_RUN_VIEW_ROOT", str(tmp_path))

    response = TestClient(app).get(f"/v1/runs/{run_id}/agents/agent-002/history")

    assert response.status_code == 200
    body = response.json()
    assert body["agent_id"] == "agent-002"
    assert body["total_records"] == 1
    assert body["records"][0]["decision"]["action"] == "watch"


def test_agent_history_sparse_trace_row_defaults(tmp_path: Path) -> None:
    run_id = "history-sparse"
    run_dir = tmp_path / "research" / "llm" / "ai-society" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "traces.jsonl").write_text(
        json.dumps({"run_id": run_id, "step": 0}) + "\n", encoding="utf-8"
    )

    history = build_agent_history(RunCatalog(tmp_path).get(run_id), "agent-000")

    assert history["total_records"] == 1
    assert history["records"][0]["timestamp"].endswith("Z")
    assert history["records"][0]["decision"] == {}
    assert history["records"][0]["tool_calls"] == []
