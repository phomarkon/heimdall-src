from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from heimdall_ai_society.config import SocietyRunConfig
from heimdall_ai_society.memory import (
    MemoryItem,
    load_memory_bank,
    memory_prompt_context,
    merge_memory_items,
    select_memory_items,
    write_memory_bank,
)
from heimdall_ai_society.reviewer import _deterministic_lessons, _hybrid_compress
from heimdall_ai_society.runner import _prompt, run_society


def test_memory_schema_validation() -> None:
    item = MemoryItem(
        scope="agent",
        lesson_key="wrong_side",
        lesson="Prefer watch when side evidence is mixed.",
        source_run_id="run-a",
        source_agent_id="agent-000",
        archetype="p2h",
        target_roles=["action_agent"],
        lesson_type="bid_guardrail",
        severity="strict",
        required_current_evidence=["both_side_probe"],
        valid_after=datetime(2026, 4, 2, 12, tzinfo=UTC),
        tags=["wrong_side", "wrong_side"],
    )
    assert item.tags == ["wrong_side"]
    assert item.target_roles == ["action_agent"]
    assert item.severity == "strict"
    assert item.valid_after.tzinfo is not None


def test_old_memory_bank_loads_with_v2_defaults(tmp_path: Path) -> None:
    bank = tmp_path / "old.jsonl"
    bank.write_text(
        json.dumps(
            {
                "scope": "agent",
                "lesson_key": "missing_authoritative_simulation",
                "lesson": "Old lesson",
                "source_run_id": "old-run",
                "source_agent_id": "agent-000",
                "archetype": "p2h",
                "valid_after": "2026-04-02T12:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    item = load_memory_bank(bank)[0]
    assert item.lesson_type == "tool_discipline"
    assert item.severity == "warning"
    assert item.target_roles == []


def test_memory_selection_caps_and_rejects_same_window_reuse() -> None:
    run_start = datetime(2026, 4, 3, 11, tzinfo=UTC)
    items = [
        _item("old-a", "agent-000", "p2h", run_start - timedelta(days=1), uses=3),
        _item("old-b", "agent-000", "p2h", run_start - timedelta(hours=1), uses=2),
        _item("same-window", "agent-000", "p2h", run_start, uses=99),
        _item("other-agent", "agent-999", "p2h", run_start - timedelta(days=1), uses=99),
    ]
    selected = select_memory_items(
        items,
        agent_id="agent-000",
        archetype="p2h",
        agent_role="action_agent",
        run_start=run_start,
        max_items_per_agent=5,
        max_prompt_chars=1000,
    )
    assert [item.lesson_key for item in selected] == ["old-a", "old-b"]


def test_memory_selection_rejects_non_matching_roles() -> None:
    run_start = datetime(2026, 4, 3, 11, tzinfo=UTC)
    items = [
        _item("action", "agent-000", "p2h", run_start - timedelta(days=1), target_roles=["action_agent"]),
        _item("risk", "agent-000", "p2h", run_start - timedelta(days=1), target_roles=["risk_officer"]),
    ]
    selected = select_memory_items(
        items,
        agent_id="agent-000",
        archetype="p2h",
        agent_role="action_agent",
        run_start=run_start,
        max_items_per_agent=5,
        max_prompt_chars=1000,
    )
    assert [item.lesson_key for item in selected] == ["action"]


def test_synthesis_roles_do_not_receive_action_only_bid_lessons() -> None:
    run_start = datetime(2026, 4, 3, 11, tzinfo=UTC)
    items = [
        _item("bid", "agent-000", "p2h", run_start - timedelta(days=1), target_roles=["risk_officer"], lesson_type="bid_guardrail"),
        _item("watch", "agent-000", "p2h", run_start - timedelta(days=1), target_roles=["risk_officer"], lesson_type="watch_calibration"),
    ]
    selected = select_memory_items(
        items,
        agent_id="agent-000",
        archetype="p2h",
        agent_role="risk_officer",
        run_start=run_start,
        max_items_per_agent=5,
        max_prompt_chars=1000,
    )
    assert [item.lesson_key for item in selected] == ["watch"]


def test_memory_merge_prunes_duplicates_and_caps() -> None:
    when = datetime(2026, 4, 2, tzinfo=UTC)
    merged = merge_memory_items(
        [_item("dup", "agent-000", "p2h", when, uses=1, severity="warning")],
        [_item("dup", "agent-000", "p2h", when, uses=2, severity="strict"), _item("keep", "agent-000", "p2h", when, uses=1)],
        max_items_per_agent=1,
    )
    assert len(merged) == 1
    assert merged[0].lesson_key == "dup"
    assert merged[0].uses == 3
    assert merged[0].severity == "strict"


def test_memory_prompt_injection_under_budget() -> None:
    item = _item("wrong_side", "agent-000", "p2h", datetime(2026, 4, 2, tzinfo=UTC))
    persona = type(
        "Persona",
        (),
        {
            "agent_id": "agent-000",
            "archetype": type("Archetype", (), {"value": "p2h"})(),
            "risk_attitude": type("Risk", (), {"value": "neutral"})(),
            "capacity_mw": 2.0,
            "storage_mwh": 0.0,
            "info_latency_min": 0,
        },
    )()
    tick = type(
        "Tick",
        (),
        {
            "timestamp": datetime(2026, 4, 3, 11, tzinfo=UTC),
            "market_price_eur_mwh": 50.0,
            "forecast": _forecast(),
        },
    )()
    messages = _prompt(
        persona,
        tick,
        objective="bid_seeking",
        ablation_strategy="diverse_action_society",
        memory_context=memory_prompt_context([item]),
    )
    payload = json.loads(messages[1]["content"])
    assert payload["memory_context"]["authority"] == "strict_role_aware_memory"
    assert "Memory cannot authorize a bid" in payload["memory_context"]["policy"]
    assert payload["memory_context"]["lessons"][0]["severity"] == "warning"


def test_dryrun_memory_disabled_has_no_loaded_memory(tmp_path: Path) -> None:
    config = SocietyRunConfig(
        run_id="memory-disabled",
        agent_count=2,
        ticks=1,
        verifier_mode="mock",
        output_dir=tmp_path,
        llm={"enabled": False},
        memory_enabled=False,
    )
    run_dir = asyncio.run(run_society(config))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    first = json.loads((run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["memory_enabled"] is False
    assert summary["memory_items_loaded"] == 0
    assert first["memory_item_count"] == 0


def test_fixture_memory_appears_in_prompt_when_enabled(tmp_path: Path) -> None:
    bank = tmp_path / "memory.jsonl"
    write_memory_bank(bank, [_item("old-lesson", "agent-000", "p2h", datetime(2025, 1, 1, tzinfo=UTC))])
    selected = select_memory_items(
        [MemoryItem.model_validate_json(bank.read_text(encoding="utf-8").splitlines()[0])],
        agent_id="agent-000",
        archetype="p2h",
        agent_role="action_agent",
        run_start=datetime(2026, 4, 2, tzinfo=UTC),
        max_items_per_agent=5,
        max_prompt_chars=1000,
    )
    assert memory_prompt_context(selected)["lessons"][0]["lesson_key"] == "old-lesson"  # type: ignore[index]


def test_review_run_creates_memory_from_synthetic_failures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    eval_dir = tmp_path / "eval-a"
    run_dir.mkdir()
    eval_dir.mkdir()
    traces = [
        {
            "run_id": "run-a",
            "step": 0,
            "timestamp": "2026-04-02T11:00:00Z",
            "agent_id": "agent-000",
            "archetype": "p2h",
            "agent_role": "opportunity_scout",
            "zone": "DK1",
            "decision": {"action": "bid"},
            "verifier_accepted": False,
            "verifier_reason_codes": ["missing_authoritative_proxy_simulation"],
        }
    ]
    (run_dir / "traces.jsonl").write_text("\n".join(json.dumps(row) for row in traces) + "\n", encoding="utf-8")
    bids = pd.DataFrame(
        [
            {
                "run_id": "run-a",
                "step": 0,
                "timestamp_utc": pd.Timestamp("2026-04-02T11:00:00Z"),
                "agent_id": "agent-000",
                "archetype": "p2h",
                "zone": "DK1",
                "side": "up",
                "quantity_mwh": 1.0,
                "limit_price_eur_mwh": 90.0,
                "verifier_accepted": False,
                "status": "wrong_side",
                "cleared_mwh": 0.0,
                "activated_volume_mwh": 0.0,
                "realized_profit_eur": 0.0,
                "profit_per_mwh": 0.0,
                "price_distance_eur_mwh": 0.0,
                "forecast_interval_covered": False,
            }
        ]
    )
    bids.to_parquet(eval_dir / "bid_evaluations.parquet", index=False)
    lessons = _deterministic_lessons(run_dir=run_dir, evaluation_dir=eval_dir)
    keys = {lesson.lesson_key for lesson in lessons}
    assert {"wrong_side", "missing_authoritative_simulation"} <= keys
    wrong_side = next(lesson for lesson in lessons if lesson.lesson_key == "wrong_side" and lesson.scope == "agent")
    missing_sim = next(lesson for lesson in lessons if lesson.lesson_key == "missing_authoritative_simulation")
    assert wrong_side.target_roles == ["opportunity_scout"]
    assert wrong_side.lesson_type == "bid_guardrail"
    assert wrong_side.severity == "strict"
    assert missing_sim.lesson_type == "tool_discipline"
    assert missing_sim.severity == "strict"


def test_bad_hybrid_reviewer_output_falls_back_safely() -> None:
    item = _item("wrong_side", "agent-000", "p2h", datetime(2026, 4, 2, tzinfo=UTC))
    assert _hybrid_compress([item], SocietyRunConfig().llm) is None


def _item(
    key: str,
    agent_id: str,
    archetype: str,
    valid_after: datetime,
    *,
    uses: int = 1,
    target_roles: list[str] | None = None,
    lesson_type: str = "bid_guardrail",
    severity: str = "warning",
) -> MemoryItem:
    return MemoryItem(
        scope="agent",
        lesson_key=key,
        lesson=f"Lesson {key}",
        source_run_id="source",
        source_agent_id=agent_id,
        archetype=archetype,
        evidence="synthetic evidence",
        tags=[key],
        target_roles=target_roles or ["action_agent"],
        lesson_type=lesson_type,
        severity=severity,
        required_current_evidence=["current evidence"],
        valid_after=valid_after,
        uses=uses,
    )


def _forecast():
    return type(
        "Forecast",
        (),
        {
            "zone": "DK1",
            "interval_for_side": lambda self, side: (90.0, 110.0) if side == "up" else (10.0, 30.0),
        },
    )()
