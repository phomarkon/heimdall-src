from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MemoryScope = Literal["agent", "archetype", "global"]
LessonType = Literal["bid_guardrail", "watch_calibration", "tool_discipline", "synthesis_guardrail", "risk_filter"]
MemorySeverity = Literal["info", "warning", "strict"]

SYNTHESIS_ROLES = frozenset(
    {
        "society_chair",
        "risk_officer",
        "explanation_editor",
        "market_mechanics_expert",
        "imbalance_analytics_expert",
        "trading_risk_monitor",
        "grid_constraint_analyst",
        "outage_impact_scorer",
        "limit_price_specialist",
        "candidate_sizing_specialist",
        "uncertainty_auditor",
        "decision_auditor",
    }
)
SYNTHESIS_LESSON_TYPES = frozenset({"watch_calibration", "synthesis_guardrail", "risk_filter"})
ACTION_LESSON_TYPES = frozenset({"bid_guardrail", "watch_calibration", "tool_discipline", "risk_filter"})
SEVERITY_RANK: dict[MemorySeverity, int] = {"info": 0, "warning": 1, "strict": 2}


class MemoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: MemoryScope
    lesson_key: str = Field(min_length=1, max_length=120)
    lesson: str = Field(min_length=1, max_length=500)
    source_run_id: str
    source_agent_id: str | None = None
    archetype: str | None = None
    evidence: str = Field(default="", max_length=500)
    tags: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    lesson_type: LessonType = "bid_guardrail"
    severity: MemorySeverity = "warning"
    required_current_evidence: list[str] = Field(default_factory=list)
    valid_after: datetime
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    uses: int = 1

    @field_validator("valid_after", "created_at_utc")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("tags")
    @classmethod
    def _dedupe_tags(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(str(tag)[:80] for tag in value if str(tag).strip()))

    @field_validator("target_roles", "required_current_evidence")
    @classmethod
    def _dedupe_short_strings(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(str(item)[:120] for item in value if str(item).strip()))


def load_memory_bank(path: Path | None) -> list[MemoryItem]:
    if path is None or not path.exists():
        return []
    items: list[MemoryItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(_load_memory_item(line))
    return items


def _load_memory_item(line: str) -> MemoryItem:
    payload = json.loads(line)
    if "target_roles" not in payload:
        payload["target_roles"] = []
    if "lesson_type" not in payload:
        payload["lesson_type"] = _legacy_lesson_type(str(payload.get("lesson_key", "")))
    if "severity" not in payload:
        payload["severity"] = "warning"
    if "required_current_evidence" not in payload:
        payload["required_current_evidence"] = []
    return MemoryItem.model_validate(payload)


def write_memory_bank(path: Path, items: list[MemoryItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8")


def append_memory_candidates(path: Path, items: list[MemoryItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(item.model_dump_json() + "\n")


def merge_memory_items(
    existing: list[MemoryItem],
    candidates: list[MemoryItem],
    *,
    max_items_per_agent: int,
) -> list[MemoryItem]:
    by_key: dict[tuple[str, str | None, str | None, str, tuple[str, ...], str], MemoryItem] = {}
    for item in [*existing, *candidates]:
        key = (
            item.scope,
            item.source_agent_id if item.scope == "agent" else None,
            item.archetype if item.scope == "archetype" else None,
            item.lesson_key,
            tuple(item.target_roles),
            item.lesson_type,
        )
        prior = by_key.get(key)
        if prior is None:
            by_key[key] = item
            continue
        strongest = prior.severity if SEVERITY_RANK[prior.severity] >= SEVERITY_RANK[item.severity] else item.severity
        evidence = item.evidence if item.created_at_utc >= prior.created_at_utc else prior.evidence
        by_key[key] = prior.model_copy(
            update={
                "uses": prior.uses + item.uses,
                "created_at_utc": max(prior.created_at_utc, item.created_at_utc),
                "severity": strongest,
                "evidence": evidence,
            }
        )
    merged = sorted(by_key.values(), key=lambda item: (SEVERITY_RANK[item.severity], item.uses, item.created_at_utc), reverse=True)
    return _cap_items(merged, max_items_per_agent=max_items_per_agent)


def select_memory_items(
    items: list[MemoryItem],
    *,
    agent_id: str,
    archetype: str,
    agent_role: str,
    run_start: datetime,
    max_items_per_agent: int,
    max_prompt_chars: int,
) -> list[MemoryItem]:
    if run_start.tzinfo is None:
        run_start = run_start.replace(tzinfo=UTC)
    else:
        run_start = run_start.astimezone(UTC)
    relevant = [
        item
        for item in items
        if item.valid_after < run_start
        and _role_matches(item, agent_role)
        and _lesson_type_allowed(item, agent_role)
        and (
            item.scope == "global"
            or (item.scope == "agent" and item.source_agent_id == agent_id)
            or (item.scope == "archetype" and item.archetype == archetype)
        )
    ]
    relevant = _cap_items(sorted(relevant, key=lambda item: (SEVERITY_RANK[item.severity], item.uses, item.created_at_utc), reverse=True), max_items_per_agent=max_items_per_agent)
    selected: list[MemoryItem] = []
    used = 0
    for item in relevant:
        size = len(_prompt_line(item))
        if used + size > max_prompt_chars:
            continue
        selected.append(item)
        used += size
    return selected


def memory_prompt_context(items: list[MemoryItem]) -> dict[str, object] | None:
    if not items:
        return None
    return {
        "authority": "strict_role_aware_memory",
        "policy": (
            "Memory cannot authorize a bid. Strict lessons must be followed unless current tools, simulator, verifier, "
            "observed_at evidence, or bidding policy contradict them. Final bids still require exact accepted simulator "
            "evidence matching side, quantity, and limit price."
        ),
        "lessons": [
            {
                "scope": item.scope,
                "lesson_key": item.lesson_key,
                "lesson_type": item.lesson_type,
                "severity": item.severity,
                "target_roles": item.target_roles,
                "lesson": item.lesson,
                "evidence": item.evidence,
                "required_current_evidence": item.required_current_evidence,
                "tags": item.tags,
            }
            for item in items
        ],
    }


def memory_audit_summary(items: list[MemoryItem]) -> list[dict[str, Any]]:
    return [
        {
            "lesson_key": item.lesson_key,
            "lesson_type": item.lesson_type,
            "severity": item.severity,
            "target_roles": item.target_roles,
        }
        for item in items
    ]


def memory_fingerprint(items: list[MemoryItem]) -> str:
    payload = [item.model_dump(mode="json", exclude={"created_at_utc"}) for item in items]
    return str(abs(hash(json.dumps(payload, sort_keys=True))))


def _prompt_line(item: MemoryItem) -> str:
    return f"{item.scope}:{item.lesson_key}:{item.lesson_type}:{item.severity}:{','.join(item.target_roles)}:{item.lesson}:{item.evidence}"


def _role_matches(item: MemoryItem, agent_role: str) -> bool:
    return not item.target_roles or agent_role in item.target_roles


def _lesson_type_allowed(item: MemoryItem, agent_role: str) -> bool:
    if agent_role in SYNTHESIS_ROLES:
        return item.lesson_type in SYNTHESIS_LESSON_TYPES
    return item.lesson_type in ACTION_LESSON_TYPES


def _legacy_lesson_type(lesson_key: str) -> LessonType:
    if lesson_key == "missing_authoritative_simulation":
        return "tool_discipline"
    if lesson_key == "missed_profitable_watch":
        return "watch_calibration"
    if lesson_key == "verifier_false_accept_review":
        return "risk_filter"
    return "bid_guardrail"


def _cap_items(items: list[MemoryItem], *, max_items_per_agent: int) -> list[MemoryItem]:
    agent_counts: dict[str, int] = {}
    archetype_counts: dict[str, int] = {}
    global_count = 0
    capped: list[MemoryItem] = []
    for item in items:
        if item.scope == "agent":
            key = item.source_agent_id or ""
            if agent_counts.get(key, 0) >= max_items_per_agent:
                continue
            agent_counts[key] = agent_counts.get(key, 0) + 1
        elif item.scope == "archetype":
            key = item.archetype or ""
            if archetype_counts.get(key, 0) >= max_items_per_agent:
                continue
            archetype_counts[key] = archetype_counts.get(key, 0) + 1
        else:
            if global_count >= max_items_per_agent:
                continue
            global_count += 1
        capped.append(item)
    return capped
