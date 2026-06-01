"""Autonomy smoke-gate for AI-society agency runs.

The agency modes (preprobe_mode in {context_only, specialist_context, none}, or cp11/cp12)
only add LLM value if the model actually calls tools itself. Before my autonomy fix
(runner._decide_with_tools forcing tool_choice="required" on the first round when no menu is
seeded), Qwen3 returned empty tool_calls under tool_choice="auto" and collapsed to all
watch/abstain — autonomous_tool_call_rate=0, zero bids, zero profit.

This gate reuses the canonical ``_tool_autonomy_metrics`` from evaluate_society_run.py and
fails (exit 1) if a run recorded ZERO autonomous (``llm_requested``) tool calls. Run it on a
2-tick agency-mode smoke BEFORE launching a full matrix so a still-collapsed agency path stops
fast instead of spending hours producing nulls.

NOTE: ``autonomous_tool_call_rate==0`` is EXPECTED and fine for the selector path
(preprobe_mode=full): there the runner pre-seeds the menu and the LLM only calls the forced
propose_action. Only gate the agency-mode smokes with this.

Usage:
    PYTHONPATH=. uv run python tools/evaluation/check_autonomy.py ai-society/runs/<run_id> [more ...]
    PYTHONPATH=. uv run python tools/evaluation/check_autonomy.py --min-rate 0.1 ai-society/runs/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_society_run as ev


def check_run(run_dir: Path, *, min_rate: float) -> tuple[bool, dict]:
    traces_path = run_dir / "traces.jsonl"
    if not traces_path.exists() or traces_path.stat().st_size == 0:
        return False, {"run_id": run_dir.name, "error": "no traces"}
    traces = ev._load_traces(traces_path)
    metrics = ev._tool_autonomy_metrics(traces)
    count = int(metrics.get("llm_tool_call_count", 0))
    rate = float(metrics.get("autonomous_tool_call_rate", 0.0) or 0.0)
    ok = count > 0 and rate >= min_rate
    return ok, {
        "run_id": run_dir.name,
        "autonomous_tool_call_rate": rate,
        "llm_tool_call_count": count,
        "tool_call_provenance_counts": metrics.get("tool_call_provenance_counts", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Run dirs holding traces.jsonl")
    parser.add_argument("--min-rate", type=float, default=0.0,
                        help="Require autonomous_tool_call_rate >= this (default 0.0: any autonomous call passes)")
    args = parser.parse_args()

    all_ok = True
    for run_dir in args.run_dirs:
        ok, info = check_run(run_dir, min_rate=args.min_rate)
        flag = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        prov = info.get("tool_call_provenance_counts", {})
        print(f"[{flag}] {info['run_id']}: autonomous={info.get('llm_tool_call_count', 0)} "
              f"rate={info.get('autonomous_tool_call_rate', 0.0)} provenance={prov}"
              + (f" ({info['error']})" if info.get("error") else ""))
    if not all_ok:
        print("\nAUTONOMY GATE FAILED: at least one agency-mode smoke recorded zero autonomous tool calls. "
              "Do not launch the full matrix; the agency path is still collapsed.", file=sys.stderr)
        return 1
    print("\nAUTONOMY GATE PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
