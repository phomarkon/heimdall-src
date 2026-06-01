from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from heimdall_ai_society.config import LLMConfig
from heimdall_ai_society.llm_client import LLMClientError, OpenAICompatibleLLMClient
from heimdall_ai_society.memory import (
    SEVERITY_RANK,
    SYNTHESIS_ROLES,
    MemoryItem,
    append_memory_candidates,
    load_memory_bank,
    merge_memory_items,
    write_memory_bank,
)
from tools.evaluation.evaluate_society_run import EvaluationInputs, evaluate_society_run


def review_run(
    *,
    run_dir: Path,
    context_dir: Path,
    truth_dir: Path,
    memory_bank: Path,
    output_dir: Path | None = None,
    reviewer_mode: str = "code_only",
    llm_config: LLMConfig | None = None,
    max_items_per_agent: int = 5,
) -> dict[str, Any]:
    output_dir = output_dir or run_dir / "memory_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = Path("evaluations") / run_dir.name
    if not (evaluation_dir / "bid_evaluations.parquet").exists():
        evaluate_society_run(EvaluationInputs(run_dir=run_dir, context_dir=context_dir, truth_dir=truth_dir, output_dir=evaluation_dir))
    candidates = _deterministic_lessons(run_dir=run_dir, evaluation_dir=evaluation_dir)
    if reviewer_mode == "hybrid_llm" and llm_config is not None:
        candidates = _hybrid_compress(candidates, llm_config) or candidates
    existing = load_memory_bank(memory_bank)
    merged = merge_memory_items(existing, candidates, max_items_per_agent=max_items_per_agent)
    write_memory_bank(memory_bank, merged)
    append_memory_candidates(output_dir / "memory_candidates.jsonl", candidates)
    review = {
        "ok": True,
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "reviewer_mode": reviewer_mode,
        "candidate_count": len(candidates),
        "memory_bank": str(memory_bank),
        "memory_bank_count": len(merged),
        "evaluation_dir": str(evaluation_dir),
        "top_lessons": [item.model_dump(mode="json") for item in candidates[:20]],
    }
    (output_dir / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return review


def _deterministic_lessons(*, run_dir: Path, evaluation_dir: Path) -> list[MemoryItem]:
    traces = _load_trace_payloads(run_dir / "traces.jsonl")
    bids = pd.read_parquet(evaluation_dir / "bid_evaluations.parquet")
    run_id = run_dir.name
    valid_after = _run_end(traces)
    roles_by_agent = _roles_by_agent(traces)
    items: list[MemoryItem] = []
    items.extend(_status_lessons(bids, run_id=run_id, valid_after=valid_after, roles_by_agent=roles_by_agent))
    items.extend(_missed_watch_lessons(traces, bids, run_id=run_id, valid_after=valid_after, roles_by_agent=roles_by_agent))
    items.extend(_tool_misuse_lessons(traces, run_id=run_id, valid_after=valid_after, roles_by_agent=roles_by_agent))
    return _dedupe_candidates(items)


def _status_lessons(bids: pd.DataFrame, *, run_id: str, valid_after: datetime, roles_by_agent: dict[str, str]) -> list[MemoryItem]:
    lessons: list[MemoryItem] = []
    templates = {
        "wrong_side": (
            "Wrong-side bids are costly: probe both up and down evidence when possible, and prefer watch unless current activation/opportunity evidence clearly supports the accepted candidate side.",
            "bid_guardrail",
            "strict",
            ["both_side_probe", "activation_or_opportunity_side_alignment", "accepted_simulator_candidate"],
        ),
        "price_not_crossed": (
            "Accepted simulator bids can still miss clearing: when current evidence is actionable, test a price ladder before finalizing the exact accepted candidate with the best clearing/profit tradeoff.",
            "bid_guardrail",
            "warning",
            ["price_ladder", "accepted_simulator_candidate", "clearing_profit_tradeoff"],
        ),
    }
    for status, (lesson, lesson_type, severity, required_evidence) in templates.items():
        subset = bids[bids["status"] == status]
        for (agent_id, archetype), group in subset.groupby(["agent_id", "archetype"]):
            agent_role = roles_by_agent.get(str(agent_id), "action_agent")
            lessons.append(
                MemoryItem(
                    scope="agent",
                    lesson_key=status,
                    lesson=lesson,
                    source_run_id=run_id,
                    source_agent_id=str(agent_id),
                    archetype=str(archetype),
                    evidence=f"{len(group)} {status} outcomes in {run_id}; latest step {int(group['step'].max())}.",
                    tags=[status, "bid_quality"],
                    target_roles=[agent_role],
                    lesson_type=lesson_type,
                    severity=severity,
                    required_current_evidence=required_evidence,
                    valid_after=valid_after,
                    uses=len(group),
                )
            )
        for archetype, group in subset.groupby("archetype"):
            roles = sorted({roles_by_agent.get(str(agent_id), "action_agent") for agent_id in group["agent_id"].dropna().unique()})
            lessons.append(
                MemoryItem(
                    scope="archetype",
                    lesson_key=status,
                    lesson=lesson,
                    source_run_id=run_id,
                    archetype=str(archetype),
                    evidence=f"{len(group)} {status} outcomes for archetype {archetype} in {run_id}.",
                    tags=[status, "bid_quality"],
                    target_roles=roles,
                    lesson_type=lesson_type,
                    severity=severity,
                    required_current_evidence=required_evidence,
                    valid_after=valid_after,
                    uses=len(group),
                )
            )
    false_accepts = bids[(bids["verifier_accepted"] == True) & (bids["realized_profit_eur"] < 0)]  # noqa: E712
    if not false_accepts.empty:
        lessons.append(
            MemoryItem(
                scope="global",
                lesson_key="verifier_false_accept_review",
                lesson="Treat any negative realized-profit accepted bid pattern as high risk; keep final bids exact simulator matches and prefer watch until inspected.",
                source_run_id=run_id,
                evidence=f"{len(false_accepts)} accepted bids had negative realized profit in {run_id}.",
                tags=["verifier", "risk"],
                target_roles=sorted(SYNTHESIS_ROLES),
                lesson_type="risk_filter",
                severity="strict",
                required_current_evidence=["negative_realized_profit_pattern", "risk_review"],
                valid_after=valid_after,
                uses=len(false_accepts),
            )
        )
        lessons.append(
            MemoryItem(
                scope="global",
                lesson_key="verifier_false_accept_review",
                lesson="Treat negative realized-profit accepted bid patterns as risk warnings; keep final bids exact simulator matches and prefer watch until current evidence is inspected.",
                source_run_id=run_id,
                evidence=f"{len(false_accepts)} accepted bids had negative realized profit in {run_id}.",
                tags=["verifier", "risk"],
                target_roles=sorted(set(roles_by_agent.values()) - SYNTHESIS_ROLES) or ["action_agent"],
                lesson_type="risk_filter",
                severity="warning",
                required_current_evidence=["exact_simulator_match", "risk_review"],
                valid_after=valid_after,
                uses=len(false_accepts),
            )
        )
    return lessons


def _missed_watch_lessons(
    traces: list[dict[str, Any]],
    bids: pd.DataFrame,
    *,
    run_id: str,
    valid_after: datetime,
    roles_by_agent: dict[str, str],
) -> list[MemoryItem]:
    profitable_ticks = set(
        tuple(row)
        for row in bids.loc[bids["status"].isin(["filled", "partially_filled"]), ["timestamp_utc", "zone"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    lessons: list[MemoryItem] = []
    missed: dict[tuple[str, str], int] = {}
    for payload in traces:
        key = (pd.Timestamp(payload["timestamp"]), payload.get("zone", "DK1"))
        decision = payload.get("decision") or {}
        if key in profitable_ticks and decision.get("action") in {"watch", "abstain"} and decision.get("watch_label") != "must_watch":
            agent_key = (str(payload.get("agent_id")), str(payload.get("archetype")))
            missed[agent_key] = missed.get(agent_key, 0) + 1
    for (agent_id, archetype), count in missed.items():
        lessons.append(
            MemoryItem(
                scope="agent",
                lesson_key="missed_profitable_watch",
                lesson="On similar high-edge MTUs, mark must_watch when profitable activation evidence is present; keep bidding separate and bid only if current simulator evidence is exact and side-aligned.",
                source_run_id=run_id,
                source_agent_id=agent_id,
                archetype=archetype,
                evidence=f"{count} profitable/filled ticks were watch or abstain without must_watch in {run_id}.",
                tags=["watch_hours", "missed_opportunity"],
                target_roles=[roles_by_agent.get(agent_id, "action_agent")],
                lesson_type="watch_calibration",
                severity="warning",
                required_current_evidence=["profitable_activation_evidence", "current_mtu_context"],
                valid_after=valid_after,
                uses=count,
            )
        )
    return lessons


def _tool_misuse_lessons(traces: list[dict[str, Any]], *, run_id: str, valid_after: datetime, roles_by_agent: dict[str, str]) -> list[MemoryItem]:
    counts: dict[tuple[str, str], int] = {}
    for payload in traces:
        decision = payload.get("decision") or {}
        if decision.get("action") != "bid" or payload.get("verifier_accepted") is not False:
            continue
        reasons = payload.get("verifier_reason_codes") or []
        if "missing_authoritative_proxy_simulation" not in reasons:
            continue
        key = (str(payload.get("agent_id")), str(payload.get("archetype")))
        counts[key] = counts.get(key, 0) + 1
    return [
        MemoryItem(
            scope="agent",
            lesson_key="missing_authoritative_simulation",
            lesson="Never finalize a bid unless the current run contains an accepted authoritative simulator/proxy call matching side, quantity, and limit exactly.",
            source_run_id=run_id,
            source_agent_id=agent_id,
            archetype=archetype,
            evidence=f"{count} rejected final bids lacked exact accepted simulator evidence in {run_id}.",
            tags=["verifier", "tool_use"],
            target_roles=[roles_by_agent.get(agent_id, "action_agent")],
            lesson_type="tool_discipline",
            severity="strict",
            required_current_evidence=["exact_accepted_simulator_match"],
            valid_after=valid_after,
            uses=count,
        )
        for (agent_id, archetype), count in counts.items()
    ]


def _hybrid_compress(items: list[MemoryItem], llm_config: LLMConfig) -> list[MemoryItem] | None:
    evidence = [item.model_dump(mode="json") for item in items[:30]]
    if not evidence:
        return items
    client = OpenAICompatibleLLMClient(
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        model=llm_config.model,
        temperature=0.0,
        max_tokens=min(llm_config.max_tokens, 1200),
        timeout_seconds=llm_config.timeout_seconds,
    )
    try:
        # The client is async-only; hybrid compression is optional, so avoid wiring
        # another event loop into the CLI in v1. Returning None triggers code-only.
        _ = client
        return None
    except (LLMClientError, json.JSONDecodeError, ValueError):
        return None


def _load_trace_payloads(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _roles_by_agent(traces: list[dict[str, Any]]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for payload in traces:
        agent_id = payload.get("agent_id")
        if agent_id is None:
            continue
        roles[str(agent_id)] = str(payload.get("agent_role") or roles.get(str(agent_id)) or "action_agent")
    return roles


def _run_end(traces: list[dict[str, Any]]) -> datetime:
    if not traces:
        return datetime.now(tz=UTC)
    end = max(pd.Timestamp(payload["timestamp"]).to_pydatetime() for payload in traces)
    if end.tzinfo is None:
        return end.replace(tzinfo=UTC)
    return end.astimezone(UTC)


def _dedupe_candidates(items: list[MemoryItem]) -> list[MemoryItem]:
    seen: set[tuple[str, str | None, str | None, str, tuple[str, ...], str]] = set()
    out: list[MemoryItem] = []
    for item in sorted(items, key=lambda value: (SEVERITY_RANK[value.severity], value.uses), reverse=True):
        key = (item.scope, item.source_agent_id, item.archetype, item.lesson_key, tuple(item.target_roles), item.lesson_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
