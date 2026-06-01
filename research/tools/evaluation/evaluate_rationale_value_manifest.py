from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.evaluation.evaluate_rationale_value import (
    case_study_table,
    load_traces,
    render_report,
    sample_pair_examples,
    score_rows,
    summarize,
    window_delta_table,
    write_csv,
)

RUN_ROOT = Path("ai-society/runs")


@dataclass(frozen=True)
class ExplicitPair:
    comparison: str
    setup: str
    window: str
    llm_run_id: str
    deterministic_run_id: str
    llm_path: Path
    deterministic_path: Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate rationale value over explicit LLM-vs-deterministic run-pair manifests."
    )
    parser.add_argument(
        "--manifest",
        choices=["chooser-20260522", "highfill-20260522", "hybrid-20260522"],
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples-per-pair", type=int, default=8)
    args = parser.parse_args()

    pairs = build_manifest(args.manifest)
    validate_pairs(pairs)

    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for pair in pairs:
        llm = load_traces(pair.llm_path)
        deterministic = load_traces(pair.deterministic_path)
        rows.extend(_score_rows(pair, "llm", llm))
        rows.extend(_score_rows(pair, "deterministic", deterministic))
        examples.extend(sample_pair_examples(_legacy_pair(pair), llm, deterministic, args.max_samples_per_pair))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "rationale_scores.csv", rows)
    summary = summarize(rows)
    dimension_deltas = dimension_delta_table_dynamic(summary)
    window_deltas = window_delta_table(rows)
    case_studies = case_study_table(examples)
    write_csv(args.output_dir / "rationale_summary.csv", summary)
    write_csv(args.output_dir / "rationale_dimension_deltas.csv", dimension_deltas)
    write_csv(args.output_dir / "rationale_window_deltas.csv", window_deltas)
    write_csv(args.output_dir / "rationale_case_studies.csv", case_studies)
    write_csv(args.output_dir / "rationale_annotation_sample.csv", examples)
    write_csv(args.output_dir / "pair_manifest.csv", [pair_payload(pair) for pair in pairs])
    (args.output_dir / "rationale_value_report.md").write_text(
        render_report(summary, dimension_deltas, window_deltas, case_studies, examples),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": args.manifest,
                "pairs": len(pairs),
                "rows": len(rows),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


def build_manifest(name: str) -> list[ExplicitPair]:
    if name == "chooser-20260522":
        return chooser_pairs()
    if name == "highfill-20260522":
        return highfill_pairs()
    if name == "hybrid-20260522":
        return hybrid_pairs()
    raise ValueError(f"unknown manifest: {name}")


def chooser_pairs() -> list[ExplicitPair]:
    pairs: list[ExplicitPair] = []
    root = RUN_ROOT / "chooser-det-llm-20260522"
    setups = ("s06-actioncore", "s12-balanced", "mixed20")
    windows = ("apr02-0530", "apr09-1830", "apr13-0015", "apr17-1900", "apr28-1900")
    for setup in setups:
        for window in windows:
            deterministic = f"cdl-{setup}-deterministic-{window}-seed42-24-q32"
            for variant in ("guarded", "shadow-toolvisible"):
                llm = f"cdl-{setup}-{variant}-{window}-seed42-24-q32"
                pairs.append(
                    explicit_pair(
                        comparison=variant,
                        setup=f"{setup}/{variant}",
                        window=window,
                        llm_run_id=llm,
                        deterministic_run_id=deterministic,
                        llm_root=root,
                        deterministic_root=root,
                    )
                )
    return pairs


def highfill_pairs() -> list[ExplicitPair]:
    pairs: list[ExplicitPair] = []
    root = RUN_ROOT / "high-fill-llm-s06-20260522"
    windows = ("apr02-0530", "apr09-1830", "apr13-0015")
    variants = ("llm-fill-selector", "llm-fill-selector-memory-v2", "llm-fill-selector-retry")
    for window in windows:
        deterministic = f"hfl-s06-actioncore-deterministic-highfill-{window}-seed42-24-q32"
        for variant in variants:
            llm = f"hfl-s06-actioncore-{variant}-{window}-seed42-24-q32"
            pairs.append(
                explicit_pair(
                    comparison=variant,
                    setup=f"s06-actioncore/{variant}",
                    window=window,
                    llm_run_id=llm,
                    deterministic_run_id=deterministic,
                    llm_root=root,
                    deterministic_root=root,
                )
            )
    return pairs


def hybrid_pairs() -> list[ExplicitPair]:
    pairs: list[ExplicitPair] = []
    deterministic_root = RUN_ROOT / "chooser-det-llm-20260522"
    windows = ("apr02-0530", "apr09-1830", "apr13-0015")
    variants = (
        ("critic", RUN_ROOT / "det-llm-critic-20260522", "dlc-s06-actioncore-critic"),
        ("retry-council", RUN_ROOT / "llm-retry-council-s06-20260522", "lrc-s06-actioncore-retry"),
        (
            "retry-council-memory-v2",
            RUN_ROOT / "memory-v2-retry-council-s06-20260522",
            "mrc-s06-actioncore-retry-memv2",
        ),
    )
    for window in windows:
        deterministic = f"cdl-s06-actioncore-deterministic-{window}-seed42-24-q32"
        for label, llm_root, prefix in variants:
            llm = f"{prefix}-{window}-seed42-24-q32"
            pairs.append(
                explicit_pair(
                    comparison=label,
                    setup=f"s06-actioncore/{label}",
                    window=window,
                    llm_run_id=llm,
                    deterministic_run_id=deterministic,
                    llm_root=llm_root,
                    deterministic_root=deterministic_root,
                )
            )
    return pairs


def explicit_pair(
    *,
    comparison: str,
    setup: str,
    window: str,
    llm_run_id: str,
    deterministic_run_id: str,
    llm_root: Path,
    deterministic_root: Path,
) -> ExplicitPair:
    return ExplicitPair(
        comparison=comparison,
        setup=setup,
        window=window,
        llm_run_id=llm_run_id,
        deterministic_run_id=deterministic_run_id,
        llm_path=llm_root / llm_run_id / "traces.jsonl",
        deterministic_path=deterministic_root / deterministic_run_id / "traces.jsonl",
    )


def validate_pairs(pairs: list[ExplicitPair]) -> None:
    if not pairs:
        raise RuntimeError("manifest produced no pairs")
    for pair in pairs:
        validate_trace(pair.llm_path)
        validate_trace(pair.deterministic_path)


def validate_trace(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    summary_path = path.parent / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = load_traces(path)
    expected = int(summary["ticks"]) * int(summary["agent_count"])
    if len(rows) != expected:
        raise RuntimeError(f"{path} row count {len(rows)} != expected {expected}")
    failures = [
        row
        for row in rows
        if "LLM call failed" in str((row.get("decision") or {}).get("rationale") or row.get("rationale") or "")
    ]
    if failures:
        raise RuntimeError(f"{path} contains {len(failures)} LLM failure fallback rationales")


def _score_rows(pair: ExplicitPair, system: str, traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = score_rows(_legacy_pair(pair), system, traces)
    for row in rows:
        row["comparison"] = pair.comparison
        row["llm_run_id"] = pair.llm_run_id
        row["deterministic_run_id"] = pair.deterministic_run_id
    return rows


def _legacy_pair(pair: ExplicitPair) -> Any:
    return type(
        "RunPairAdapter",
        (),
        {
            "setup": pair.setup,
            "window": pair.window,
            "llm_run_id": pair.llm_run_id,
            "det_run_id": pair.deterministic_run_id,
        },
    )()


def dimension_delta_table_dynamic(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_setup = {(row["setup"], row["system"]): row for row in summary}
    setups = [*sorted({row["setup"] for row in summary if row["setup"] != "ALL"}), "ALL"]
    rows = []
    for setup in setups:
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
                "total_delta": _delta(llm, det, "mean_rationale_value_0_16"),
                "specificity_delta": _delta(llm, det, "mean_specificity_0_4"),
                "actionability_delta": _delta(llm, det, "mean_actionability_0_4"),
                "faithfulness_proxy_delta": _delta(llm, det, "mean_faithfulness_proxy_0_4"),
                "contrastiveness_delta": _delta(llm, det, "mean_contrastiveness_0_4"),
                "evidence_category_delta": _delta(llm, det, "mean_evidence_categories"),
                "hallucinated_acceptance_rate_delta": _nullable_delta(
                    llm["hallucinated_acceptance_rate"], det["hallucinated_acceptance_rate"]
                ),
                "selected_bid_support_rate_llm": llm["selected_bid_support_rate"],
                "selected_bid_support_rate_deterministic": det["selected_bid_support_rate"],
            }
        )
    return rows


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float:
    return round(float(left[key]) - float(right[key]), 3)


def _nullable_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 3)


def pair_payload(pair: ExplicitPair) -> dict[str, str]:
    return {
        "comparison": pair.comparison,
        "setup": pair.setup,
        "window": pair.window,
        "llm_run_id": pair.llm_run_id,
        "deterministic_run_id": pair.deterministic_run_id,
        "llm_trace": str(pair.llm_path),
        "deterministic_trace": str(pair.deterministic_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
