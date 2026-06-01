from __future__ import annotations

from heimdall_ai_society.schemas import ToolCallRecord


def _empty_critic_totals() -> dict[str, object]:
    return {
        "review_count": 0,
        "keep_count": 0,
        "veto_count": 0,
        "veto_reason_counts": {},
        "mutation_attempt_count": 0,
        "forecast_disagreement_veto_count": 0,
    }


def _accumulate_critic_totals(totals: dict[str, object], *, records: list[ToolCallRecord]) -> None:
    for record in records:
        if record.name != "llm_critic_review":
            continue
        result = record.result
        totals["review_count"] = int(totals["review_count"]) + 1
        outcome = str(result.get("critic_outcome", ""))
        if outcome in {"keep_bid", "ignored_mutation", "no_valid_critic_decision"}:
            totals["keep_count"] = int(totals["keep_count"]) + 1
        if outcome in {"veto_to_watch", "veto_to_abstain"}:
            totals["veto_count"] = int(totals["veto_count"]) + 1
        if result.get("mutation_attempt"):
            totals["mutation_attempt_count"] = int(totals["mutation_attempt_count"]) + 1
        if result.get("forecast_disagreement_veto"):
            totals["forecast_disagreement_veto_count"] = int(totals["forecast_disagreement_veto_count"]) + 1
        reason_counts = totals["veto_reason_counts"]
        if not isinstance(reason_counts, dict):
            reason_counts = {}
            totals["veto_reason_counts"] = reason_counts
        for reason in result.get("veto_reasons", []) or []:
            key = str(reason)
            reason_counts[key] = int(reason_counts.get(key, 0)) + 1


def _finalize_critic_totals(totals: dict[str, object]) -> dict[str, object]:
    reviews = max(1, int(totals["review_count"]))
    return {
        **totals,
        "veto_rate": round(int(totals["veto_count"]) / reviews, 6),
        "mutation_attempt_rate": round(int(totals["mutation_attempt_count"]) / reviews, 6),
    }


def _empty_fill_selector_totals() -> dict[str, object]:
    return {
        "review_count": 0,
        "candidate_count": 0,
        "select_count": 0,
        "watch_count": 0,
        "abstain_count": 0,
        "mutation_attempt_count": 0,
        "no_accepted_candidate_count": 0,
        "selected_clear_probability_proxy_values": [],
        "selected_worst_case_profit_values": [],
    }


def _accumulate_fill_selector_totals(totals: dict[str, object], *, records: list[ToolCallRecord]) -> None:
    for record in records:
        if record.name != "fill_selector_review":
            continue
        result = record.result
        totals["review_count"] = int(totals["review_count"]) + 1
        outcome = str(result.get("selector_outcome", ""))
        if outcome == "select_candidate":
            totals["select_count"] = int(totals["select_count"]) + 1
        if outcome == "watch":
            totals["watch_count"] = int(totals["watch_count"]) + 1
        if outcome == "abstain":
            totals["abstain_count"] = int(totals["abstain_count"]) + 1
        if outcome == "no_accepted_candidate":
            totals["no_accepted_candidate_count"] = int(totals["no_accepted_candidate_count"]) + 1
        if result.get("mutation_attempt"):
            totals["mutation_attempt_count"] = int(totals["mutation_attempt_count"]) + 1
        totals["candidate_count"] = int(totals["candidate_count"]) + int(result.get("candidate_count", 0) or 0)
        clear_values = totals["selected_clear_probability_proxy_values"]
        profit_values = totals["selected_worst_case_profit_values"]
        if isinstance(clear_values, list) and result.get("selected_clear_probability_proxy") is not None:
            clear_values.append(float(result["selected_clear_probability_proxy"]))
        if isinstance(profit_values, list) and result.get("selected_worst_case_profit_eur") is not None:
            profit_values.append(float(result["selected_worst_case_profit_eur"]))


def _finalize_fill_selector_totals(totals: dict[str, object]) -> dict[str, object]:
    reviews = max(1, int(totals["review_count"]))
    return {
        **totals,
        "select_rate": round(int(totals["select_count"]) / reviews, 6),
        "watch_rate": round(int(totals["watch_count"]) / reviews, 6),
        "mutation_attempt_rate": round(int(totals["mutation_attempt_count"]) / reviews, 6),
        "selected_clear_probability_proxy_mean": _mean_or_none(totals["selected_clear_probability_proxy_values"]),
        "selected_worst_case_profit_eur_mean": _mean_or_none(totals["selected_worst_case_profit_values"]),
    }


def _mean_or_none(values: object) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 6)
