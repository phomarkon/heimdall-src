from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.evaluation.evaluate_society_run import EvaluationInputs, evaluate_society_run


def test_evaluator_scores_deterministic_bid_outcomes_and_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    context_dir = tmp_path / "data" / "real_context" / "april"
    truth_dir = tmp_path / "data" / "evaluation_truth" / "april"
    output_dir = tmp_path / "evaluations" / "run-a"
    run_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    truth_dir.mkdir(parents=True)
    (context_dir / "manifest.json").write_text(json.dumps({"visibility": "agent_context", "dataset_id": "april"}) + "\n")
    (truth_dir / "truth_manifest.json").write_text(
        json.dumps({"visibility": "evaluation_only", "dataset_id": "april"}) + "\n"
    )

    timestamp = "2026-04-01T00:00:00Z"
    trace_rows = [
        _trace("agent-a", timestamp, "up", 4.0, 70.0, verifier_accepted=True),
        _trace("agent-b", timestamp, "up", 4.0, 75.0, verifier_accepted=True),
        _trace("agent-c", timestamp, "up", 3.0, 120.0, verifier_accepted=False),
        _trace("agent-d", timestamp, "down", 2.0, 20.0, verifier_accepted=True),
        _trace("agent-e", timestamp, None, None, None, action="abstain", verifier_accepted=None),
        _trace("agent-f", timestamp, "up", 1.0, 65.0, verifier_accepted=True, submitted_at_utc="2026-03-31T23:30:01Z"),
        _trace("agent-g", timestamp, "up", -1.0, 65.0, verifier_accepted=None),
    ]
    (run_dir / "traces.jsonl").write_text("\n".join(json.dumps(row) for row in trace_rows) + "\n")

    pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp(timestamp),
                "zone": "DK1",
                "activation_direction": "up",
                "activated_volume_mwh": 6.0,
                "spot_price_eur_mwh": 50.0,
                "settlement_price_eur_mwh": 80.0,
            }
        ]
    ).to_parquet(truth_dir / "activation_truth.parquet", index=False)

    result = evaluate_society_run(
        EvaluationInputs(run_dir=run_dir, context_dir=context_dir, truth_dir=truth_dir, output_dir=output_dir)
    )

    assert result["ok"] is True
    bids = pd.read_parquet(output_dir / "bid_evaluations.parquet").sort_values("agent_id")
    statuses = dict(zip(bids["agent_id"], bids["status"], strict=True))
    assert statuses == {
        "agent-a": "filled",
        "agent-b": "partially_filled",
        "agent-c": "price_not_crossed",
        "agent-d": "wrong_side",
        "agent-e": "abstain",
        "agent-f": "gate_closed",
        "agent-g": "invalid",
    }
    assert float(bids.loc[bids["agent_id"] == "agent-a", "cleared_mwh"].iloc[0]) == pytest.approx(4.0)
    assert float(bids.loc[bids["agent_id"] == "agent-b", "cleared_mwh"].iloc[0]) == pytest.approx(2.0)
    assert float(bids["realized_profit_eur"].sum()) == pytest.approx(180.0)

    summary = json.loads((output_dir / "run_summary.json").read_text())
    assert summary["realized_profit_eur"] == pytest.approx(180.0)
    assert summary["cleared_mwh"] == pytest.approx(6.0)
    assert summary["oracle_feasible_profit_scope"] == "submitted_bid_conditional"
    assert summary["truth_window_oracle_profit_eur"] == pytest.approx(180.0)
    assert summary["profitable_watch_or_bid_recall"] == pytest.approx(1.0)
    assert summary["status_counts"]["filled"] == 1
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["trace_sha256"]
    assert manifest["context_manifest_sha256"]
    assert manifest["truth_manifest_sha256"]
    assert manifest["artifacts"]["archetype_metrics"]


def test_evaluator_reports_independent_oracle_for_watch_only_window(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-watch"
    context_dir = tmp_path / "data" / "real_context" / "april"
    truth_dir = tmp_path / "data" / "evaluation_truth" / "april"
    output_dir = tmp_path / "evaluations" / "run-watch"
    run_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    truth_dir.mkdir(parents=True)
    (context_dir / "manifest.json").write_text(json.dumps({"visibility": "agent_context", "dataset_id": "april"}) + "\n")
    (truth_dir / "truth_manifest.json").write_text(json.dumps({"visibility": "evaluation_only", "dataset_id": "april"}) + "\n")

    timestamp = "2026-04-01T00:00:00Z"
    (run_dir / "traces.jsonl").write_text(
        json.dumps(_trace("agent-a", timestamp, None, None, None, action="watch", verifier_accepted=None)) + "\n"
    )
    pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp(timestamp),
                "zone": "DK1",
                "activation_direction": "up",
                "activated_volume_mwh": 10.0,
                "spot_price_eur_mwh": 50.0,
                "settlement_price_eur_mwh": 90.0,
            }
        ]
    ).to_parquet(truth_dir / "activation_truth.parquet", index=False)

    evaluate_society_run(
        EvaluationInputs(run_dir=run_dir, context_dir=context_dir, truth_dir=truth_dir, output_dir=output_dir)
    )

    summary = json.loads((output_dir / "run_summary.json").read_text())
    assert summary["oracle_feasible_profit_eur"] == 0.0
    assert summary["truth_window_oracle_profit_eur"] == pytest.approx(400.0)
    assert summary["missed_truth_window_oracle_profit_eur"] == pytest.approx(400.0)
    assert summary["profitable_watch_or_bid_recall"] == pytest.approx(1.0)


def test_evaluator_rejects_context_dataset_as_truth(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    context_dir = tmp_path / "context"
    truth_dir = tmp_path / "truth"
    for path in [run_dir, context_dir, truth_dir]:
        path.mkdir()
    (run_dir / "traces.jsonl").write_text("")
    (context_dir / "manifest.json").write_text(json.dumps({"visibility": "agent_context"}) + "\n")
    (truth_dir / "truth_manifest.json").write_text(json.dumps({"visibility": "agent_context"}) + "\n")
    pd.DataFrame().to_parquet(truth_dir / "activation_truth.parquet")

    with pytest.raises(RuntimeError, match="expected 'evaluation_only'"):
        evaluate_society_run(EvaluationInputs(run_dir=run_dir, context_dir=context_dir, truth_dir=truth_dir, output_dir=tmp_path / "out"))


def test_evaluator_handles_no_activation_truth(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    context_dir = tmp_path / "context"
    truth_dir = tmp_path / "truth"
    for path in [run_dir, context_dir, truth_dir]:
        path.mkdir()
    (context_dir / "manifest.json").write_text(json.dumps({"visibility": "agent_context"}) + "\n")
    (truth_dir / "truth_manifest.json").write_text(json.dumps({"visibility": "evaluation_only"}) + "\n")
    timestamp = "2026-04-01T00:00:00Z"
    (run_dir / "traces.jsonl").write_text(json.dumps(_trace("agent-a", timestamp, "up", 1.0, 10.0)) + "\n")
    pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp(timestamp),
                "zone": "DK1",
                "activation_direction": "neutral",
                "activated_volume_mwh": 0.0,
                "spot_price_eur_mwh": 50.0,
                "settlement_price_eur_mwh": 50.0,
            }
        ]
    ).to_parquet(truth_dir / "activation_truth.parquet", index=False)

    evaluate_society_run(EvaluationInputs(run_dir=run_dir, context_dir=context_dir, truth_dir=truth_dir, output_dir=tmp_path / "out"))

    bids = pd.read_parquet(tmp_path / "out" / "bid_evaluations.parquet")
    assert bids["status"].tolist() == ["no_activation"]


def test_evaluator_counts_p2h_proxy_vs_real_comparison(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-p2h-compare"
    context_dir = tmp_path / "data" / "real_context"
    truth_dir = tmp_path / "data" / "truth"
    output_dir = tmp_path / "evaluations" / "run-p2h-compare"
    run_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    truth_dir.mkdir(parents=True)
    (context_dir / "manifest.json").write_text(json.dumps({"visibility": "agent_context"}) + "\n")
    (truth_dir / "truth_manifest.json").write_text(json.dumps({"visibility": "evaluation_only"}) + "\n")
    timestamp = "2026-04-01T00:00:00Z"
    trace = _trace("agent-p2h", timestamp, "up", 0.25, 80.0, verifier_accepted=True)
    trace["archetype"] = "p2h"
    trace["tool_calls"] = [
        {
            "name": "simulate_bid",
            "arguments": {"side": "up", "quantity_mwh": 0.25, "limit_price_eur_mwh": 80.0},
            "ok": True,
            "result": {
                "accepted": True,
                "backend": "proxy",
                "comparison": {
                    "proxy": {"accepted": True, "worst_case_profit_eur": 10.0},
                    "pypsa_background": {"accepted": False, "worst_case_profit_eur": -5.0},
                    "accepted_disagreement": True,
                    "proxy_false_positive": True,
                },
            },
        }
    ]
    (run_dir / "traces.jsonl").write_text(json.dumps(trace) + "\n")
    pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp(timestamp),
                "zone": "DK1",
                "activation_direction": "up",
                "activated_volume_mwh": 1.0,
                "spot_price_eur_mwh": 50.0,
                "settlement_price_eur_mwh": 90.0,
            }
        ]
    ).to_parquet(truth_dir / "activation_truth.parquet", index=False)

    evaluate_society_run(
        EvaluationInputs(run_dir=run_dir, context_dir=context_dir, truth_dir=truth_dir, output_dir=output_dir)
    )

    summary = json.loads((output_dir / "run_summary.json").read_text())
    assert summary["asset_backend_comparison_by_archetype"]["p2h"]["proxy_false_positive_count"] == 1
    assert summary["asset_backend_pypsa_background_accepted_count"] == 0


def test_evaluator_counts_three_current_backend_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-three-backend-compare"
    context_dir = tmp_path / "data" / "real_context"
    truth_dir = tmp_path / "data" / "truth"
    output_dir = tmp_path / "evaluations" / "run-three-backend-compare"
    run_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    truth_dir.mkdir(parents=True)
    (context_dir / "manifest.json").write_text(json.dumps({"visibility": "agent_context"}) + "\n")
    (truth_dir / "truth_manifest.json").write_text(json.dumps({"visibility": "evaluation_only"}) + "\n")
    timestamp = "2026-04-01T00:00:00Z"
    trace = _trace("agent-p2h", timestamp, "up", 0.25, 80.0, verifier_accepted=True)
    trace["archetype"] = "p2h"
    trace["tool_calls"] = [
        {
            "name": "simulate_bid",
            "arguments": {"side": "up", "quantity_mwh": 0.25, "limit_price_eur_mwh": 80.0},
            "ok": True,
            "result": {
                "accepted": True,
                "backend": "scenario_envelope",
                "comparison": {
                    "proxy": {"accepted": True, "worst_case_profit_eur": 10.0},
                    "scenario_envelope": {"accepted": True, "worst_case_profit_eur": 8.0},
                    "pypsa_background": {"accepted": False, "worst_case_profit_eur": -5.0},
                    "accepted_disagreement": True,
                    "proxy_false_positive": True,
                    "scenario_envelope_false_positive": True,
                },
            },
        }
    ]
    (run_dir / "traces.jsonl").write_text(json.dumps(trace) + "\n")
    pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp(timestamp),
                "zone": "DK1",
                "activation_direction": "up",
                "activated_volume_mwh": 1.0,
                "spot_price_eur_mwh": 50.0,
                "settlement_price_eur_mwh": 90.0,
            }
        ]
    ).to_parquet(truth_dir / "activation_truth.parquet", index=False)

    evaluate_society_run(
        EvaluationInputs(run_dir=run_dir, context_dir=context_dir, truth_dir=truth_dir, output_dir=output_dir)
    )

    summary = json.loads((output_dir / "run_summary.json").read_text())
    archetype = summary["asset_backend_comparison_by_archetype"]["p2h"]
    assert summary["asset_backend_scenario_envelope_accepted_count"] == 1
    assert summary["asset_backend_pypsa_background_accepted_count"] == 0
    assert summary["asset_backend_scenario_envelope_false_positive_count"] == 1
    assert archetype["mean_scenario_envelope_worst_case_profit_eur"] == 8.0
    assert archetype["mean_pypsa_background_worst_case_profit_eur"] == -5.0


def test_evaluator_counts_tool_autonomy_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-tool-autonomy"
    context_dir = tmp_path / "data" / "real_context"
    truth_dir = tmp_path / "data" / "truth"
    output_dir = tmp_path / "evaluations" / "run-tool-autonomy"
    run_dir.mkdir(parents=True)
    context_dir.mkdir(parents=True)
    truth_dir.mkdir(parents=True)
    (context_dir / "manifest.json").write_text(json.dumps({"visibility": "agent_context"}) + "\n")
    (truth_dir / "truth_manifest.json").write_text(json.dumps({"visibility": "evaluation_only"}) + "\n")
    timestamp = "2026-04-01T00:00:00Z"
    trace = _trace("agent-p2h", timestamp, "up", 0.25, 80.0, verifier_accepted=True)
    trace["tool_calls"] = [
        {
            "name": "get_activation_context",
            "arguments": {"hours": 24},
            "ok": True,
            "result": {"watch_score": 0.5},
            "provenance": "runner_seeded",
        },
        {
            "name": "simulate_bid",
            "arguments": {"side": "up", "quantity_mwh": 0.25, "limit_price_eur_mwh": 80.0},
            "ok": True,
            "result": {"accepted": True},
            "provenance": "llm_requested",
        },
        {
            "name": "propose_action",
            "arguments": {"action": "bid", "side": "up", "quantity_mwh": 0.25, "limit_price_eur_mwh": 80.0},
            "ok": True,
            "result": {},
            "provenance": "forced_final",
        },
    ]
    (run_dir / "traces.jsonl").write_text(json.dumps(trace) + "\n")
    pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp(timestamp),
                "zone": "DK1",
                "activation_direction": "up",
                "activated_volume_mwh": 1.0,
                "spot_price_eur_mwh": 50.0,
                "settlement_price_eur_mwh": 90.0,
            }
        ]
    ).to_parquet(truth_dir / "activation_truth.parquet", index=False)

    evaluate_society_run(
        EvaluationInputs(run_dir=run_dir, context_dir=context_dir, truth_dir=truth_dir, output_dir=output_dir)
    )

    summary = json.loads((output_dir / "run_summary.json").read_text())
    assert summary["tool_call_provenance_counts"]["llm_requested"] == 1
    assert summary["simulator_self_call_rate"] == 1.0
    assert summary["accepted_bid_backed_by_llm_requested_simulator_rate"] == 1.0
    assert summary["final_action_forced_rate"] == 1.0


def _trace(
    agent_id: str,
    timestamp: str,
    side: str | None,
    quantity_mwh: float | None,
    limit_price_eur_mwh: float | None,
    *,
    action: str = "bid",
    verifier_accepted: bool | None = True,
    submitted_at_utc: str | None = None,
) -> dict:
    decision = {
        "action": action,
        "side": side,
        "quantity_mwh": quantity_mwh,
        "limit_price_eur_mwh": limit_price_eur_mwh,
        "rationale": "synthetic",
        "confidence": 0.8,
    }
    if submitted_at_utc is not None:
        decision["submitted_at_utc"] = submitted_at_utc
    return {
        "run_id": "run-a",
        "step": 0,
        "timestamp": timestamp,
        "agent_id": agent_id,
        "zone": "DK1",
        "archetype": "synthetic",
        "llm_id": "dry",
        "forecaster_id": "f0",
        "decision": decision,
        "verifier_mode": "mock",
        "verifier_accepted": verifier_accepted,
        "verifier_reason_codes": [],
        "market_price_eur_mwh": 50.0,
        "forecast_interval_eur_mwh": [45.0, 85.0],
        "rationale": "synthetic",
        "unavailable_reason": None,
        "tool_calls": [],
    }
