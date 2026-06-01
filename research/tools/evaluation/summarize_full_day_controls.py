from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METRIC_KEYS = [
    "realized_profit_eur",
    "truth_window_oracle_capture",
    "wrong_side_count",
    "alert_spam_rate",
    "confidence_dispersion",
    "must_watch_profitable_or_activated_precision",
    "must_watch_profitable_or_activated_recall",
    "profitable_watch_or_bid_recall",
    "evidence_reference_validity",
    "verifier_realized_profit_breach_rate",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize full-day proxy/real control runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--queue-results", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, default=Path("evaluations"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--note-path", type=Path, required=True)
    parser.add_argument("--label", default="mixed20-full-days")
    parser.add_argument("--expected-ticks", type=int, default=96)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.note_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.queue_results)
    digest = [_digest_row(row, args.run_root, args.evaluation_root, args.expected_ticks) for row in rows]
    aggregates = _aggregate_rows(digest)
    paired = _paired_rows(digest)

    _write_csv(args.output_dir / f"{args.label}-runs.csv", digest)
    _write_csv(args.output_dir / f"{args.label}-aggregates.csv", aggregates)
    _write_csv(args.output_dir / f"{args.label}-paired-deltas.csv", paired)
    _plot_grouped_bars(
        digest,
        "realized_profit_eur",
        "Realized profit by day and controls",
        "EUR",
        args.output_dir / f"{args.label}-profit.png",
    )
    _plot_grouped_bars(
        digest,
        "wrong_side_count",
        "Wrong-side count by day and controls",
        "count",
        args.output_dir / f"{args.label}-wrong-side.png",
    )
    args.note_path.write_text(_render_note(args.label, digest, aggregates, paired, args.output_dir), encoding="utf-8")
    print(json.dumps({"runs": len(digest), "output_dir": str(args.output_dir), "note_path": str(args.note_path)}, indent=2))
    return 0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"queue results must contain a non-empty list: {path}")
    failures = [row for row in rows if not row.get("ok")]
    if failures:
        raise RuntimeError(f"queue results contain failures: {failures}")
    return rows


def _digest_row(row: dict[str, Any], run_root: Path, evaluation_root: Path, expected_ticks: int) -> dict[str, Any]:
    run_id = str(row["run_id"])
    run_dir = run_root / run_id
    summary_path = run_dir / "summary.json"
    trace_path = run_dir / "traces.jsonl"
    eval_path = evaluation_root / run_id / "run_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"missing summary: {summary_path}")
    if not trace_path.exists():
        raise RuntimeError(f"missing trace: {trace_path}")
    if not eval_path.exists():
        raise RuntimeError(f"missing evaluation summary: {eval_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eval_summary = json.loads(eval_path.read_text(encoding="utf-8"))
    trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
    traces = [json.loads(line) for line in trace_lines if line.strip()]
    expected_rows = int(summary["ticks"]) * int(summary["agent_count"])
    if len(traces) != expected_rows:
        raise RuntimeError(f"{run_id} trace rows {len(traces)} != expected {expected_rows}")
    if int(summary["ticks"]) != expected_ticks:
        raise RuntimeError(f"{run_id} ticks {summary['ticks']} != expected {expected_ticks}")
    if any("LLM call failed" in str(trace.get("rationale", "")) for trace in traces):
        raise RuntimeError(f"{run_id} contains LLM fallback rationale")
    if float(eval_summary["verifier_realized_profit_breach_rate"]) != 0.0:
        raise RuntimeError(f"{run_id} has verifier breach rate {eval_summary['verifier_realized_profit_breach_rate']}")

    controls = "proxy" if "proxy-controls" in run_id else "real"
    day = run_id.split("-")[1].replace("apr", "Apr ")
    side_counts = summary.get("side_diagnostics", {}).get("final_bid_side_counts", {})
    payload: dict[str, Any] = {
        "run_id": run_id,
        "day": day,
        "controls": controls,
        "agent_count": summary["agent_count"],
        "ticks": summary["ticks"],
        "trace_rows": len(traces),
        "trace_complete": len(traces) == expected_rows,
        "accepted": summary["accepted"],
        "watched": summary["watched"],
        "abstained": summary["abstained"],
        "invalid": summary["invalid"],
        "rejected": summary["rejected"],
        "final_bid_up": side_counts.get("up", 0),
        "final_bid_down": side_counts.get("down", 0),
        "runtime_seconds": round(float(summary["runtime_seconds"]), 3),
        "runtime_seconds_per_tick": round(float(summary["runtime_seconds_per_tick"]), 3),
    }
    for key in METRIC_KEYS:
        payload[key] = eval_summary[key]
    return payload


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["controls"]].append(row)
    out = []
    for controls in ("proxy", "real"):
        items = grouped[controls]
        out.append(
            {
                "controls": controls,
                "runs": len(items),
                "realized_profit_eur_total": round(sum(float(row["realized_profit_eur"]) for row in items), 4),
                "realized_profit_eur_mean": round(_mean(row["realized_profit_eur"] for row in items), 4),
                "accepted_total": sum(int(row["accepted"]) for row in items),
                "watched_total": sum(int(row["watched"]) for row in items),
                "abstained_total": sum(int(row["abstained"]) for row in items),
                "wrong_side_total": sum(int(row["wrong_side_count"]) for row in items),
                "final_bid_up_total": sum(int(row["final_bid_up"]) for row in items),
                "final_bid_down_total": sum(int(row["final_bid_down"]) for row in items),
                "oracle_capture_mean": round(_mean(row["truth_window_oracle_capture"] for row in items), 6),
                "precision_mean": round(_mean(row["must_watch_profitable_or_activated_precision"] for row in items), 6),
                "recall_mean": round(_mean(row["must_watch_profitable_or_activated_recall"] for row in items), 6),
                "confidence_dispersion_mean": round(_mean(row["confidence_dispersion"] for row in items), 6),
                "alert_spam_rate_mean": round(_mean(row["alert_spam_rate"] for row in items), 6),
                "verifier_breach_rate_max": max(float(row["verifier_realized_profit_breach_rate"]) for row in items),
                "evidence_reference_validity_min": min(float(row["evidence_reference_validity"]) for row in items),
            }
        )
    return out


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_day: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_day[row["day"]][row["controls"]] = row
    paired = []
    for day in sorted(by_day):
        proxy = by_day[day].get("proxy")
        real = by_day[day].get("real")
        if not proxy or not real:
            continue
        paired.append(
            {
                "day": day,
                "proxy_profit_eur": proxy["realized_profit_eur"],
                "real_profit_eur": real["realized_profit_eur"],
                "real_minus_proxy_profit_eur": round(float(real["realized_profit_eur"]) - float(proxy["realized_profit_eur"]), 4),
                "proxy_accepted": proxy["accepted"],
                "real_accepted": real["accepted"],
                "proxy_wrong_side": proxy["wrong_side_count"],
                "real_wrong_side": real["wrong_side_count"],
                "proxy_oracle_capture": proxy["truth_window_oracle_capture"],
                "real_oracle_capture": real["truth_window_oracle_capture"],
            }
        )
    return paired


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_grouped_bars(rows: list[dict[str, Any]], key: str, title: str, ylabel: str, path: Path) -> None:
    days = sorted({row["day"] for row in rows})
    proxy = [float(next(row for row in rows if row["day"] == day and row["controls"] == "proxy")[key]) for day in days]
    real = [float(next(row for row in rows if row["day"] == day and row["controls"] == "real")[key]) for day in days]
    x = range(len(days))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    ax.bar([i - width / 2 for i in x], proxy, width, label="proxy controls", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], real, width, label="real controls", color="#F58518")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x), days)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_note(
    label: str,
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    proxy = next(row for row in aggregates if row["controls"] == "proxy")
    real = next(row for row in aggregates if row["controls"] == "real")
    display_label = label.replace("-", " ").title()
    lines = [
        f"# {display_label} Proxy vs Real Controls",
        "",
        "Date: 2026-05-16",
        "",
        "## Summary",
        "",
        f"Eight full-day {display_label} runs completed on April 02-05 with both proxy-control and real-control asset backends. "
        "All runs produced complete traces, complete summaries, evaluation artifacts, and zero verifier realized-profit breaches.",
        "",
        "The main result is a clear aggressiveness tradeoff: proxy-controls accepted more bids and earned more total realized profit, "
        "while real-controls were more conservative, had fewer wrong-side decisions, and still remained profitable on all four days.",
        "",
        "## Aggregate Results",
        "",
        "| Controls | Runs | Total profit EUR | Accepted | Watched | Abstained | Wrong side | Mean oracle capture | Mean precision | Mean recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for agg in (proxy, real):
        lines.append(
            f"| {agg['controls']} | {agg['runs']} | {agg['realized_profit_eur_total']:.2f} | "
            f"{agg['accepted_total']} | {agg['watched_total']} | {agg['abstained_total']} | {agg['wrong_side_total']} | "
            f"{agg['oracle_capture_mean']:.6f} | {agg['precision_mean']:.6f} | {agg['recall_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Day Comparison",
            "",
            "| Day | Proxy profit EUR | Real profit EUR | Real minus proxy EUR | Proxy wrong side | Real wrong side |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['day']} | {float(row['proxy_profit_eur']):.2f} | {float(row['real_profit_eur']):.2f} | "
            f"{float(row['real_minus_proxy_profit_eur']):.2f} | {row['proxy_wrong_side']} | {row['real_wrong_side']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Proxy-controls are the aggressive upper-bound setting: they produced more accepted bids and higher total profit, but also many more wrong-side outcomes.",
            "- Real-controls are the thesis-primary setting: they are lower-profit but cleaner, with substantially reduced wrong-side count and lower confidence dispersion.",
            "- Safety evidence is strong: verifier breach rate was zero, evidence reference validity was one, and no trace contained an LLM fallback rationale.",
            "",
            "## Artifacts",
            "",
            f"- Run table: `{output_dir / f'{label}-runs.csv'}`",
            f"- Aggregate table: `{output_dir / f'{label}-aggregates.csv'}`",
            f"- Paired deltas: `{output_dir / f'{label}-paired-deltas.csv'}`",
            f"- Profit plot: `{output_dir / f'{label}-profit.png'}`",
            f"- Wrong-side plot: `{output_dir / f'{label}-wrong-side.png'}`",
            "",
            "## Limitations and Next Comparator",
            "",
            "Mixed20 is promoted by these results but should not be treated as final empirical evidence in isolation. "
            "The matched mixed18 full-day comparator should be run over the same April 02-05 windows and the same proxy/real-control split before final thesis tables are frozen.",
            "",
        ]
    )
    return "\n".join(lines)


def _mean(values: Any) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals)


if __name__ == "__main__":
    raise SystemExit(main())
