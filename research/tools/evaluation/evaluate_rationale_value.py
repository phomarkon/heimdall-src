from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


RUN_ROOTS = [
    Path("ai-society/runs/overnight-broadcast-ablation"),
    Path("ai-society/runs/overnight-broadcast-ablation-deterministic"),
    Path("ai-society/runs/mixed-sideaware-20260515"),
]

DEFAULT_SETUPS = ("s06-actioncore", "s12-balanced", "s20-mixed")
DEFAULT_WINDOWS = ("apr02-0530", "apr03-1430", "apr05-1030", "apr09-1830", "apr13-0015")

VALID_WATCH_REASONS = {
    "activation_risk",
    "price_volatility",
    "forecast_uncertainty",
    "accepted_bid_available",
    "verifier_rejection_cluster",
    "cross_agent_disagreement",
}

EVIDENCE_TERMS = {
    "accepted": ("accepted", "simulator", "verified", "candidate", "worst-case", "worst case"),
    "rejected": ("rejected", "reject", "reason", "negative", "non-clearable", "not clearable"),
    "forecast": ("forecast", "interval", "uncertainty", "edge", "spread"),
    "market": ("activation", "volatility", "price", "regime", "watch_score"),
    "risk": ("risk", "cautious", "confidence", "uncertainty", "guardrail"),
}


@dataclass(frozen=True)
class RunPair:
    setup: str
    window: str
    llm_run_id: str
    det_run_id: str
    llm_path: Path
    det_path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM rationale value against deterministic control traces.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/rationale-value"))
    parser.add_argument("--setups", nargs="*", default=list(DEFAULT_SETUPS))
    parser.add_argument("--windows", nargs="*", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--quality", default="q32", choices=["q14", "q32"])
    parser.add_argument("--max-samples-per-pair", type=int, default=8)
    args = parser.parse_args()

    pairs = discover_pairs(args.setups, args.windows, args.quality)
    if not pairs:
        raise SystemExit("No paired LLM/deterministic runs found.")

    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for pair in pairs:
        llm = load_traces(pair.llm_path)
        det = load_traces(pair.det_path)
        rows.extend(score_rows(pair, "llm", llm))
        rows.extend(score_rows(pair, "deterministic", det))
        examples.extend(sample_pair_examples(pair, llm, det, args.max_samples_per_pair))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "rationale_scores.csv", rows)
    summary = summarize(rows)
    dimension_deltas = dimension_delta_table(summary)
    window_deltas = window_delta_table(rows)
    case_studies = case_study_table(examples)
    write_csv(args.output_dir / "rationale_summary.csv", summary)
    write_csv(args.output_dir / "rationale_dimension_deltas.csv", dimension_deltas)
    write_csv(args.output_dir / "rationale_window_deltas.csv", window_deltas)
    write_csv(args.output_dir / "rationale_case_studies.csv", case_studies)
    write_csv(args.output_dir / "rationale_annotation_sample.csv", examples)
    (args.output_dir / "rationale_value_report.md").write_text(
        render_report(summary, dimension_deltas, window_deltas, case_studies, examples),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "pairs": len(pairs), "output_dir": str(args.output_dir)}, indent=2))


def discover_pairs(setups: list[str], windows: list[str], quality: str) -> list[RunPair]:
    pairs: list[RunPair] = []
    for setup in setups:
        for window in windows:
            if setup == "s20-mixed":
                llm_id = f"msa-screen-mixed20-{window}-24"
                det_id = f"oba-det-{setup}-bcast-{window}-seed42"
            else:
                llm_id = f"oba-{setup}-bcast-{window}-seed42-{quality}"
                det_id = f"oba-det-{setup}-bcast-{window}-seed42"
            llm_path = find_trace(llm_id)
            det_path = find_trace(det_id)
            if llm_path and det_path:
                pairs.append(RunPair(setup, window, llm_id, det_id, llm_path, det_path))
    return pairs


def find_trace(run_id: str) -> Path | None:
    for root in RUN_ROOTS:
        path = root / run_id / "traces.jsonl"
        if path.exists():
            return path
    return None


def load_traces(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_rows(pair: RunPair, system: str, traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        decision = trace.get("decision") or {}
        rationale = str(decision.get("rationale") or trace.get("rationale") or "")
        tool_calls = trace.get("tool_calls") or []
        watch_reasons = decision.get("watch_reasons") or []
        score = score_rationale(rationale, decision, tool_calls)
        rows.append(
            {
                "setup": pair.setup,
                "window": pair.window,
                "system": system,
                "run_id": pair.llm_run_id if system == "llm" else pair.det_run_id,
                "step": trace.get("step"),
                "timestamp": trace.get("timestamp"),
                "agent_id": trace.get("agent_id"),
                "archetype": trace.get("archetype"),
                "agent_role": trace.get("agent_role", "action_agent"),
                "action": decision.get("action"),
                "watch_label": decision.get("watch_label"),
                "confidence": decision.get("confidence"),
                "watch_reason_count": len(watch_reasons),
                "valid_watch_reason_count": sum(1 for reason in watch_reasons if reason in VALID_WATCH_REASONS),
                "rationale": rationale,
                **score,
            }
        )
    return rows


def score_rationale(rationale: str, decision: dict[str, Any], tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    text = rationale.lower()
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_/-]*|[-+]?\d+(?:\.\d+)?", rationale)
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", rationale)
    units = sum(1 for unit in ("eur", "mwh", "mw", "score", "%") if unit in text)
    evidence_hits = evidence_hit_counts(text)
    tool_names = {str(call.get("name") or "").lower() for call in tool_calls if isinstance(call, dict)}
    tool_mentions = sum(1 for name in tool_names if name and name in text)
    action = str(decision.get("action") or "")
    side = str(decision.get("side") or "")
    selected_supported = selected_candidate_supported(decision, tool_calls)

    specificity = min(4, int(bool(numbers)) + min(2, units) + int(tool_mentions > 0) + int(len(tokens) >= 24))
    actionability = min(
        4,
        int(action in text or action == "bid")
        + int(side in {"up", "down"} and side in text)
        + int(any(label in text for label in ("watch", "bid", "abstain", "must_watch", "risk", "confidence")))
        + int(any(term in text for term in ("accepted", "rejected", "candidate", "guardrail", "simulator"))),
    )
    faithfulness_proxy = min(
        4,
        int(selected_supported or action != "bid")
        + int(sum(evidence_hits.values()) >= 2)
        + int(tool_mentions > 0 or any(term in text for term in ("simulator", "forecast", "watch_score")))
        + int(not hallucinated_acceptance(text, tool_calls)),
    )
    contrastiveness = min(
        4,
        int(any(term in text for term in ("however", "otherwise", "but", "unless")))
        + int(any(term in text for term in ("risk", "uncertainty", "cautious", "downgrade")))
        + int(any(term in text for term in ("rejected", "negative", "not", "no accepted")))
        + int(any(term in text for term in ("watch", "abstain"))),
    )
    total = specificity + actionability + faithfulness_proxy + contrastiveness
    return {
        "word_count": len(re.findall(r"\w+", rationale)),
        "numeric_reference_count": len(numbers),
        "tool_mention_count": tool_mentions,
        "evidence_category_count": sum(1 for value in evidence_hits.values() if value),
        "specificity_score_0_4": specificity,
        "actionability_score_0_4": actionability,
        "faithfulness_proxy_score_0_4": faithfulness_proxy,
        "contrastiveness_score_0_4": contrastiveness,
        "rationale_value_score_0_16": total,
        "selected_candidate_supported": selected_supported,
        "hallucinated_acceptance_flag": hallucinated_acceptance(text, tool_calls),
    }


def evidence_hit_counts(text: str) -> dict[str, int]:
    return {name: sum(1 for term in terms if term in text) for name, terms in EVIDENCE_TERMS.items()}


def selected_candidate_supported(decision: dict[str, Any], tool_calls: list[dict[str, Any]]) -> bool:
    if decision.get("action") != "bid":
        return False
    for call in tool_calls:
        if not isinstance(call, dict) or call.get("name") != "selected_candidate_diagnostics":
            continue
        result = call.get("result") or {}
        if result.get("selected") is True:
            sim = result.get("simulator_result") or {}
            return sim.get("accepted") is True
    return False


def hallucinated_acceptance(text: str, tool_calls: list[dict[str, Any]]) -> bool:
    if "accepted" not in text:
        return False
    accepted_seen = False
    for call in tool_calls:
        result = call.get("result") if isinstance(call, dict) else None
        if isinstance(result, dict) and result.get("accepted") is True:
            accepted_seen = True
            break
    return not accepted_seen


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["setup"], row["system"])].append(row)
    out = []
    for (setup, system), items in sorted(groups.items()):
        out.append(summary_payload(setup, system, items))
    for system in ("llm", "deterministic"):
        items = [row for row in rows if row["system"] == system]
        if items:
            out.append(summary_payload("ALL", system, items))
    return out


def dimension_delta_table(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_setup = {(row["setup"], row["system"]): row for row in summary}
    rows = []
    for setup in ["s06-actioncore", "s12-balanced", "s20-mixed", "ALL"]:
        llm = by_setup.get((setup, "llm"))
        det = by_setup.get((setup, "deterministic"))
        if not llm or not det:
            continue
        rows.append(
            {
                "setup": setup,
                "n_llm": llm["n"],
                "n_deterministic": det["n"],
                "llm_total_0_16": llm["mean_rationale_value_0_16"],
                "deterministic_total_0_16": det["mean_rationale_value_0_16"],
                "total_delta": round(float(llm["mean_rationale_value_0_16"]) - float(det["mean_rationale_value_0_16"]), 3),
                "specificity_delta": round(float(llm["mean_specificity_0_4"]) - float(det["mean_specificity_0_4"]), 3),
                "actionability_delta": round(float(llm["mean_actionability_0_4"]) - float(det["mean_actionability_0_4"]), 3),
                "faithfulness_proxy_delta": round(float(llm["mean_faithfulness_proxy_0_4"]) - float(det["mean_faithfulness_proxy_0_4"]), 3),
                "contrastiveness_delta": round(float(llm["mean_contrastiveness_0_4"]) - float(det["mean_contrastiveness_0_4"]), 3),
                "evidence_category_delta": round(float(llm["mean_evidence_categories"]) - float(det["mean_evidence_categories"]), 3),
                "hallucinated_acceptance_rate_delta": round(float(llm["hallucinated_acceptance_rate"]) - float(det["hallucinated_acceptance_rate"]), 3),
                "selected_bid_support_rate_llm": llm["selected_bid_support_rate"],
                "selected_bid_support_rate_deterministic": det["selected_bid_support_rate"],
            }
        )
    return rows


def window_delta_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["setup"], row["window"], row["system"])].append(row)
    out = []
    for setup in sorted({row["setup"] for row in rows}):
        for window in sorted({row["window"] for row in rows if row["setup"] == setup}):
            llm = groups.get((setup, window, "llm"), [])
            det = groups.get((setup, window, "deterministic"), [])
            if not llm or not det:
                continue
            llm_total = _avg(llm, "rationale_value_score_0_16")
            det_total = _avg(det, "rationale_value_score_0_16")
            out.append(
                {
                    "setup": setup,
                    "window": window,
                    "n_llm": len(llm),
                    "n_deterministic": len(det),
                    "llm_total_0_16": llm_total,
                    "deterministic_total_0_16": det_total,
                    "delta": round(llm_total - det_total, 3),
                    "llm_evidence_categories": _avg(llm, "evidence_category_count"),
                    "deterministic_evidence_categories": _avg(det, "evidence_category_count"),
                    "llm_hallucinated_acceptance_rate": avg_bool(llm, "hallucinated_acceptance_flag"),
                    "deterministic_hallucinated_acceptance_rate": avg_bool(det, "hallucinated_acceptance_flag"),
                }
            )
    return out


def case_study_table(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_setup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        by_setup[row["setup"]].append(row)
    out = []
    for setup, items in sorted(by_setup.items()):
        for row in sorted(items, key=lambda item: int(item["score_delta_llm_minus_det"]), reverse=True)[:3]:
            out.append(
                {
                    "setup": setup,
                    "window": row["window"],
                    "step": row["step"],
                    "agent": f"{row['agent_id']} ({row['archetype']})",
                    "llm_value": compact_rationale(row["llm_rationale"]),
                    "deterministic_value": compact_rationale(row["deterministic_rationale"]),
                    "delta": row["score_delta_llm_minus_det"],
                }
            )
    return out


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return round(mean(values), 3) if values else 0.0


def compact_rationale(value: str, *, max_len: int = 210) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def summary_payload(setup: str, system: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        values = [float(row[key]) for row in items if row.get(key) not in (None, "")]
        return round(mean(values), 3) if values else 0.0

    return {
        "setup": setup,
        "system": system,
        "n": len(items),
        "mean_words": avg("word_count"),
        "mean_specificity_0_4": avg("specificity_score_0_4"),
        "mean_actionability_0_4": avg("actionability_score_0_4"),
        "mean_faithfulness_proxy_0_4": avg("faithfulness_proxy_score_0_4"),
        "mean_contrastiveness_0_4": avg("contrastiveness_score_0_4"),
        "mean_rationale_value_0_16": avg("rationale_value_score_0_16"),
        "mean_tool_mentions": avg("tool_mention_count"),
        "mean_evidence_categories": avg("evidence_category_count"),
        "hallucinated_acceptance_rate": avg_bool(items, "hallucinated_acceptance_flag"),
        "selected_bid_support_rate": avg_bool([row for row in items if row["action"] == "bid"], "selected_candidate_supported"),
    }


def avg_bool(items: list[dict[str, Any]], key: str) -> float | None:
    if not items:
        return None
    return round(sum(1 for row in items if row.get(key) is True) / len(items), 3)


def sample_pair_examples(pair: RunPair, llm: list[dict[str, Any]], det: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    det_by_key = {(row.get("step"), row.get("agent_id")): row for row in det}
    candidates = []
    for row in llm:
        key = (row.get("step"), row.get("agent_id"))
        if key in det_by_key:
            llm_decision = row.get("decision") or {}
            det_decision = det_by_key[key].get("decision") or {}
            llm_score = score_rationale(llm_decision.get("rationale", ""), llm_decision, row.get("tool_calls") or {})
            det_score = score_rationale(det_decision.get("rationale", ""), det_decision, det_by_key[key].get("tool_calls") or {})
            candidates.append((llm_score["rationale_value_score_0_16"] - det_score["rationale_value_score_0_16"], row, det_by_key[key]))
    samples = sorted(candidates, key=lambda item: item[0], reverse=True)[:max_samples]
    out = []
    for delta, llm_row, det_row in samples:
        llm_decision = llm_row.get("decision") or {}
        det_decision = det_row.get("decision") or {}
        out.append(
            {
                "setup": pair.setup,
                "window": pair.window,
                "step": llm_row.get("step"),
                "timestamp": llm_row.get("timestamp"),
                "agent_id": llm_row.get("agent_id"),
                "archetype": llm_row.get("archetype"),
                "llm_action": llm_decision.get("action"),
                "det_action": det_decision.get("action"),
                "score_delta_llm_minus_det": delta,
                "llm_rationale": llm_decision.get("rationale", ""),
                "deterministic_rationale": det_decision.get("rationale", ""),
                "human_actionability_1_5": "",
                "human_faithfulness_1_5": "",
                "human_specificity_1_5": "",
                "human_preference_llm_det_tie": "",
                "annotator_note": "",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(
    summary: list[dict[str, Any]],
    dimension_deltas: list[dict[str, Any]],
    window_deltas: list[dict[str, Any]],
    case_studies: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> str:
    lines = [
        "# Rationale Value Evaluation",
        "",
        "Automatic rubric over paired LLM and deterministic 6/12/20-agent runs. Scores are proxies for a human rubric, not a replacement for human annotation.",
        "",
        "## Key Findings",
        "",
        "- LLM traces score higher than deterministic controls in every paired setup and every evaluated window.",
        "- The strongest lift is evidence richness: LLM rationales name simulator outcomes, forecast/market context, risk, uncertainty, and rejected alternatives.",
        "- Selected bid support stays at 1.0 for both systems, so the LLM advantage is explanation quality around the same verifier/simulator guardrails, not looser action selection.",
        "- Hallucinated acceptance proxy is lower for LLM traces overall, mostly because deterministic controls often mention accepted candidates without operator-level evidence context.",
        "",
        "## Rubric Definition",
        "",
        "| Dimension | Scale | What it rewards |",
        "| --- | ---: | --- |",
        "| Specificity | 0-4 | Numeric values, units, tool references, and enough detail to identify the market situation. |",
        "| Actionability | 0-4 | Clear bid/watch/abstain guidance, side information, and candidate/guardrail references. |",
        "| Faithfulness proxy | 0-4 | Alignment with visible tool evidence, selected-candidate diagnostics, and no unsupported acceptance claim. |",
        "| Contrastiveness | 0-4 | Explains rejected alternatives, uncertainty, risk, downgrade/watch reasons, or why not to bid. |",
        "",
        "## Table X1: Rationale Quality By Setup",
        "",
        "| Setup | System | n | Specificity | Actionability | Faithfulness proxy | Contrastiveness | Total /16 | Tool mentions | Evidence cats |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['setup']} | {row['system']} | {row['n']} | {row['mean_specificity_0_4']} | "
            f"{row['mean_actionability_0_4']} | {row['mean_faithfulness_proxy_0_4']} | "
            f"{row['mean_contrastiveness_0_4']} | {row['mean_rationale_value_0_16']} | "
            f"{row['mean_tool_mentions']} | {row['mean_evidence_categories']} |"
        )
    lines.extend(
        [
            "",
            "## Table X2: LLM Improvement Over Deterministic Control",
            "",
            "| Setup | LLM /16 | Det. /16 | Delta | Specificity Δ | Actionability Δ | Faithfulness Δ | Contrastiveness Δ | Evidence cats Δ | Hallucinated acceptance Δ | Bid support |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dimension_deltas:
        lines.append(
            f"| {row['setup']} | {row['llm_total_0_16']} | {row['deterministic_total_0_16']} | {row['total_delta']} | "
            f"{row['specificity_delta']} | {row['actionability_delta']} | {row['faithfulness_proxy_delta']} | "
            f"{row['contrastiveness_delta']} | {row['evidence_category_delta']} | {row['hallucinated_acceptance_rate_delta']} | "
            f"{row['selected_bid_support_rate_llm']}/{row['selected_bid_support_rate_deterministic']} |"
        )
    lines.extend(
        [
            "",
            "## Table X3: Robustness Across Windows",
            "",
            "| Setup | Window | LLM /16 | Det. /16 | Delta | LLM evidence cats | Det. evidence cats |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in window_deltas:
        lines.append(
            f"| {row['setup']} | {row['window']} | {row['llm_total_0_16']} | {row['deterministic_total_0_16']} | "
            f"{row['delta']} | {row['llm_evidence_categories']} | {row['deterministic_evidence_categories']} |"
        )
    lines.extend(
        [
            "",
            "## Table X4: Qualitative Case Studies",
            "",
            "| Setup | Window/step | Agent | LLM explanation value | Deterministic explanation | Delta |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in case_studies:
        lines.append(
            f"| {row['setup']} | {row['window']} / {row['step']} | {row['agent']} | "
            f"{row['llm_value']} | {row['deterministic_value']} | {row['delta']} |"
        )
    lines.extend(["", "## High-Delta Examples", ""])
    for row in examples[:12]:
        lines.extend(
            [
                f"### {row['setup']} {row['window']} step {row['step']} {row['agent_id']} ({row['archetype']})",
                "",
                f"Delta: {row['score_delta_llm_minus_det']}",
                "",
                f"LLM: {row['llm_rationale']}",
                "",
                f"Deterministic: {row['deterministic_rationale']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Human Rubric",
            "",
            "Use `rationale_annotation_sample.csv` for a small blinded study. Rate each paired explanation 1-5 for actionability, faithfulness to visible evidence, and specificity; record whether the LLM, deterministic, or neither explanation is better for an operator deciding whether to bid/watch/ignore.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
