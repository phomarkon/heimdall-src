from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PAIRS = {
    "safetyctx-s06-actioncore-apr02-0530-seed13-q32": "fco-s06-actioncore-bcast-apr02-0530-seed13-q32",
    "safetyctx-s06-actioncore-apr02-0530-seed137-q32": "fco-s06-actioncore-bcast-apr02-0530-seed137-q32",
    "safetyctx-s20-mixed-apr09-1830-seed13-q32": "fco-s20-mixed-bcast-apr09-1830-seed13-q32",
    "safetyctx-s20-mixed-apr09-1830-seed137-q32": "fco-s20-mixed-bcast-apr09-1830-seed137-q32",
}

METRICS = [
    "realized_profit_eur",
    "truth_window_oracle_capture",
    "bid_action_count",
    "wrong_side_count",
    "max_drawdown_eur",
    "downside_cvar_95_eur",
]


def main() -> None:
    output_dir = Path("evaluations/safety-context-only-unguarded-20260519")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    examples = []
    for unguarded, guarded in PAIRS.items():
        guarded_summary = _read_json(Path("evaluations") / guarded / "run_summary.json")
        unguarded_summary = _read_json(Path("evaluations") / unguarded / "run_summary.json")
        trace_rows = _read_traces(Path("ai-society/runs/safety-context-only-unguarded-20260519") / unguarded / "traces.jsonl")
        shadow = _shadow_metrics(trace_rows)
        rows.append(
            {
                "unguarded_run_id": unguarded,
                "guarded_run_id": guarded,
                "metrics": {
                    key: {
                        "guarded": guarded_summary.get(key),
                        "unguarded": unguarded_summary.get(key),
                        "delta": _delta(unguarded_summary.get(key), guarded_summary.get(key)),
                    }
                    for key in METRICS
                },
                **shadow,
            }
        )
        examples.extend(_shadow_rejection_examples(unguarded, trace_rows))
    payload = {"pairs": rows, "shadow_rejection_examples": examples[:20]}
    (output_dir / "paired_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_traces(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _delta(left: Any, right: Any) -> float | None:
    try:
        return round(float(left) - float(right), 6)
    except (TypeError, ValueError):
        return None


def _shadow_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    shadows = []
    for row in rows:
        for call in row.get("tool_calls") or []:
            if call.get("name") == "shadow_required_simulation":
                shadows.append(call.get("result") or {})
    rejected = [result for result in shadows if result.get("shadow_accepted") is not True]
    negative_worst = [
        result
        for result in shadows
        if _float_or_none(result.get("shadow_worst_case_profit_eur")) is not None
        and _float_or_none(result.get("shadow_worst_case_profit_eur")) < 0
    ]
    return {
        "shadow_bid_count": len(shadows),
        "would_have_been_blocked_count": len(rejected),
        "shadow_reject_rate": round(len(rejected) / len(shadows), 6) if shadows else None,
        "shadow_negative_worst_case_count": len(negative_worst),
        "shadow_negative_worst_case_rate": round(len(negative_worst) / len(shadows), 6) if shadows else None,
    }


def _shadow_rejection_examples(run_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        decision = row.get("decision") or {}
        if decision.get("action") != "bid":
            continue
        for call in row.get("tool_calls") or []:
            if call.get("name") != "shadow_required_simulation":
                continue
            result = call.get("result") or {}
            if result.get("shadow_accepted") is True:
                continue
            examples.append(
                {
                    "run_id": run_id,
                    "step": row.get("step"),
                    "timestamp": row.get("timestamp"),
                    "agent_id": row.get("agent_id"),
                    "archetype": row.get("archetype"),
                    "decision": decision,
                    "shadow_reason_codes": result.get("shadow_reason_codes", []),
                    "shadow_worst_case_profit_eur": result.get("shadow_worst_case_profit_eur"),
                }
            )
    return examples


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
