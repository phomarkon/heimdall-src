"""Priority signal computation for timeline highlighting."""

from __future__ import annotations

import math
from typing import Any

from heimdall_run_view._utils import RunContext, _float


def _priority_signals(
    total_steps: int,
    rows_by_step: dict[int, list[dict[str, Any]]],
    context: RunContext,
) -> dict[int, dict[str, Any]]:
    raw = {
        step: _raw_priority_signal(
            step, rows_by_step.get(step, []), context.bid_rows_by_step.get(step, [])
        )
        for step in range(total_steps)
    }
    realized_by_step: dict[int, float] = {}
    attempted_steps: set[int] = set()
    for step in range(total_steps):
        eval_rows = context.bid_rows_by_step.get(step, [])
        realized_by_step[step] = sum(_float(r.get("realized_profit_eur"), 0.0) for r in eval_rows)
        bid_outcome = any(
            str(r.get("status") or "")
            in {"filled", "partially_filled", "price_not_crossed", "wrong_side"}
            or _float(r.get("cleared_mwh"), 0.0) > 0
            for r in eval_rows
        )
        bid_decision = any(
            _decision_action_inline(row) == "bid" for row in rows_by_step.get(step, [])
        )
        if bid_outcome or bid_decision:
            attempted_steps.add(step)

    if context.bid_rows_by_step:
        captured = {step: value for step, value in realized_by_step.items() if value > 0}
        ranked_captured = sorted(captured, key=lambda step: captured[step], reverse=True)
        n_cap = len(ranked_captured)
        crit_cut = max(1, round(n_cap * 0.25))
        high_cut = max(crit_cut, round(n_cap * 0.6))
        captured_tier = {
            step: ("critical" if i < crit_cut else "high" if i < high_cut else "medium")
            for i, step in enumerate(ranked_captured)
        }
        for step in range(total_steps):
            signal = raw[step]
            signal["grounding"] = "realized_outcome"
            realized = realized_by_step[step]
            if step in captured_tier:
                signal["tier"] = captured_tier[step]
                signal["score"] = round(realized, 2)
                signal["drivers"] = [f"captured EUR {round(realized)}", *signal["drivers"][:3]]
            elif step in attempted_steps:
                signal["tier"] = "watch"
                signal["score"] = 0.0
                signal["drivers"] = ["bids attempted, none filled", *signal["drivers"][:3]]
            else:
                signal["tier"] = "low"
                signal["score"] = 0.0
        ranked = sorted(range(total_steps), key=lambda step: realized_by_step[step], reverse=True)
        count = max(1, total_steps)
        for rank, step in enumerate(ranked, start=1):
            raw[step]["rank"] = rank
            raw[step]["percentile"] = round(1.0 - (rank - 1) / count, 4)
        labels = {
            "critical": "Top value captured",
            "high": "Value captured",
            "medium": "Minor value captured",
            "watch": "Attempted, no fill",
            "low": "Quiet",
        }
    else:
        max_expected = max(
            1.0,
            *(
                max(0.0, _float(s["components"].get("expected_profit_eur"), 0.0))
                for s in raw.values()
            ),
        )
        for signal in raw.values():
            components = signal["components"]
            signal["grounding"] = "forward_estimate"
            normalized_expected = (
                max(0.0, _float(components.get("expected_profit_eur"), 0.0)) / max_expected
            )
            components["normalized_expected_profit"] = round(normalized_expected, 3)
            score = 1.3 * normalized_expected
            score += 0.6 if components.get("accepted_bid_available") else 0.0
            score += 0.5 * _float(components.get("side_consensus"), 0.0)
            score += 0.4 * _float(components.get("activation_score"), 0.0)
            score -= 0.4 * _float(components.get("price_not_clearable_risk"), 0.0)
            signal["score"] = round(max(0.0, score), 3)
        ranked = sorted(raw.items(), key=lambda item: item[1]["score"], reverse=True)
        count = max(1, len(ranked))
        for rank, (step, signal) in enumerate(ranked, start=1):
            percentile = 1.0 - ((rank - 1) / count)
            signal["rank"] = rank
            signal["percentile"] = round(percentile, 4)
            signal["tier"] = _priority_tier(signal, rank, percentile, count)
        labels = {
            "critical": "Critical (predicted)",
            "high": "High (predicted)",
            "medium": "Medium (predicted)",
            "watch": "Watch evidence",
            "low": "Low priority",
        }

    for signal in raw.values():
        signal["label"] = labels[signal["tier"]]
    return raw


def _raw_priority_signal(
    step: int, rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    bid_rows = [row for row in rows if _decision_action_inline(row) == "bid"]
    sides = [str((row.get("decision") or {}).get("side") or "") for row in bid_rows]
    accepted_candidates = 0
    accepted_sides: list[str] = []
    positive_worst_case = 0
    expected_profit = 0.0
    worst_case_profit = 0.0
    clearability = 0.0
    must_watch_count = 0
    watch_count = 0
    activation_hits = 0
    mismatch_risk_hits = 0
    negative_worst_case_rejections = 0
    price_not_clearable_hits = 0
    rejection_count = 0
    watch_score = 0.0
    drivers: list[str] = []
    risks: list[str] = []

    for row in rows:
        decision = row.get("decision") or {}
        side = str(decision.get("side") or "")
        watch_label = str(decision.get("watch_label") or "")
        if watch_label == "must_watch":
            must_watch_count += 1
        if watch_label in {"watch", "must_watch"} or _decision_action_inline(row) == "watch":
            watch_count += 1
            activation_hits += 1
        for call in row.get("tool_calls") or []:
            result = call.get("result") or {}
            arguments = call.get("arguments") or {}
            reason_codes = [str(code) for code in result.get("reason_codes") or []]
            accepted = result.get("accepted")
            if accepted is False:
                rejection_count += 1
            mismatch_risk_hits += sum(
                1
                for code in reason_codes
                if "mismatch" in code or "wrong_side" in code or "activation" in code
            )
            price_not_clearable_hits += sum(
                1
                for code in reason_codes
                if "price_not" in code or "clearable" in code or "not_clear" in code
            )
            call_side = str(arguments.get("side") or side or "")
            if accepted is True:
                accepted_candidates += 1
                if call_side:
                    accepted_sides.append(call_side)
            value = _first_number(
                result,
                "worst_case_profit_eur",
                "rough_worst_case_profit_eur",
                "worst_case_profit_proxy_eur",
            )
            if value is not None:
                worst_case_profit = max(worst_case_profit, value)
                positive_worst_case += int(value > 0)
                if accepted is False and value < 0:
                    negative_worst_case_rejections += 1
            expected = _first_number(
                result,
                "rough_expected_profit_eur",
                "expected_profit_proxy_eur",
                "expected_spread_eur_mwh",
            )
            if expected is not None:
                expected_profit = max(expected_profit, expected)
            clear_prob = _first_number(result, "clear_probability_proxy", "score")
            if clear_prob is not None:
                clearability = max(clearability, max(0.0, min(1.0, clear_prob)))
            watch = _first_number(result, "watch_score")
            if watch is not None:
                watch_score = max(watch_score, max(0.0, min(1.0, watch)))
            signals = result.get("signals") or {}
            hint = signals.get("activation_direction_hint") or result.get("direction_hint")
            if hint in {"up", "down"}:
                activation_hits += int(hint in sides or hint in accepted_sides)
                mismatch_risk_hits += int(
                    bool(sides or accepted_sides) and hint not in sides + accepted_sides
                )

    accepted_bid_available = (
        any(row.get("verifier_accepted") is True for row in bid_rows) or accepted_candidates > 0
    )
    must_watch_share = must_watch_count / max(1, len(rows))
    watch_share = watch_count / max(1, len(rows))
    side_consensus = _largest_share(sides + accepted_sides)
    side_disagreement = len({side for side in sides + accepted_sides if side}) > 1
    filled_count = sum(
        1
        for row in eval_rows
        if str(row.get("status") or "") == "filled" or _float(row.get("cleared_mwh"), 0.0) > 0
    )
    realized_profit = sum(_float(row.get("realized_profit_eur"), 0.0) for row in eval_rows)
    filled_share = filled_count / max(1, len(eval_rows) or len(rows))
    watch_activation_hint = max(watch_score, watch_share, must_watch_share)
    activation_score = min(1.0, filled_share + 0.25 * watch_activation_hint)
    clearability = max(clearability, min(1.0, filled_count / max(1, len(eval_rows) or len(rows))))
    risk_denominator = max(1, rejection_count + accepted_candidates)
    price_not_clearable_risk = min(1.0, price_not_clearable_hits / risk_denominator)
    side_activation_mismatch_risk = min(
        1.0, (mismatch_risk_hits + int(side_disagreement)) / max(1, len(rows))
    )
    negative_worst_case_rejection_risk = min(1.0, negative_worst_case_rejections / risk_denominator)

    if accepted_bid_available:
        drivers.append("accepted candidate")
    if must_watch_share > 0:
        drivers.append("must-watch share")
    if side_consensus >= 0.65:
        drivers.append("same-side consensus")
    if activation_score >= 0.4:
        drivers.append("activation score")
    if expected_profit > 0 or worst_case_profit > 0:
        drivers.append("positive simulator edge")
    if side_disagreement:
        risks.append("side disagreement")
    if side_activation_mismatch_risk > 0:
        risks.append("side/activation mismatch")
    if price_not_clearable_risk > 0:
        risks.append("price-not-clearable risk")
    if negative_worst_case_rejection_risk > 0:
        risks.append("negative worst-case rejection")

    return {
        "score": 0.0,
        "rank": None,
        "percentile": 0.0,
        "tier": "low",
        "label": "Low priority",
        "drivers": drivers[:4],
        "risks": risks[:4],
        "components": {
            "must_watch_share": round(must_watch_share, 3),
            "watch_share": round(watch_share, 3),
            "accepted_bid_available": accepted_bid_available,
            "side_consensus": round(side_consensus, 3),
            "activation_score": round(activation_score, 3),
            "clearability": round(clearability, 3),
            "price_not_clearable_risk": round(price_not_clearable_risk, 3),
            "side_activation_mismatch_risk": round(side_activation_mismatch_risk, 3),
            "negative_worst_case_rejection_risk": round(negative_worst_case_rejection_risk, 3),
            "expected_profit_eur": round(expected_profit, 2),
            "worst_case_profit_eur": round(worst_case_profit, 2),
            "realized_profit_eur": round(realized_profit, 2),
        },
    }


def _priority_tier(signal: dict[str, Any], rank: int, percentile: float, count: int) -> str:
    components = signal.get("components") or {}
    has_watch_evidence = (
        _float(components.get("watch_share"), 0.0) > 0
        or _float(components.get("must_watch_share"), 0.0) > 0
    )
    critical_cutoff = max(5, math.ceil(count * 0.10))
    high_cutoff = max(8, math.ceil(count * 0.20))
    medium_cutoff = max(12, math.ceil(count * 0.25))
    if rank <= critical_cutoff:
        return "critical"
    if rank <= high_cutoff:
        return "high"
    if rank <= medium_cutoff:
        return "medium"
    if has_watch_evidence:
        return "watch"
    return "low"


def _priority_accuracy(
    priority_by_step: dict[int, dict[str, Any]], context: RunContext
) -> dict[str, Any]:
    selected = {
        step for step, signal in priority_by_step.items() if signal["tier"] in {"critical", "high"}
    }
    positives = {
        step
        for step, rows in context.bid_rows_by_step.items()
        if any(
            _float(row.get("realized_profit_eur"), 0.0) > 0
            or _float(row.get("cleared_mwh"), 0.0) > 0
            for row in rows
        )
    }
    true_positive = len(selected & positives)
    precision = true_positive / len(selected) if selected else 0.0
    recall = true_positive / len(positives) if positives else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    total_profit = sum(
        max(0.0, _float(row.get("realized_profit_eur"), 0.0))
        for rows in context.bid_rows_by_step.values()
        for row in rows
    )
    selected_profit = sum(
        max(0.0, _float(row.get("realized_profit_eur"), 0.0))
        for step, rows in context.bid_rows_by_step.items()
        if step in selected
        for row in rows
    )
    return {
        "score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "profit_capture_rate": round(selected_profit / total_profit, 4) if total_profit else 0.0,
        "selected_tick_count": len(selected),
        "positive_tick_count": len(positives),
    }


def _largest_share(values: list[str]) -> float:
    filtered = [value for value in values if value]
    if not filtered:
        return 0.0
    return max(filtered.count(value) for value in set(filtered)) / len(filtered)


def _first_number(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


# Inlined decision_action to avoid circular import with _trace
def _decision_action_inline(row: dict[str, Any]) -> str:
    return str((row.get("decision") or {}).get("action") or "abstain")
