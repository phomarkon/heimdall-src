from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from heimdall_ai_society.config import load_config
from heimdall_ai_society.reviewer import review_run
from heimdall_ai_society.runner import run_society


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="heimdall_ai_society")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="Parse a society config without side effects.")
    validate.add_argument("config", type=Path)

    run = subparsers.add_parser("run", help="Run a small society simulation.")
    run.add_argument("--config", required=True, type=Path)

    review = subparsers.add_parser("review-run", help="Review a completed run and merge between-run memory lessons.")
    review.add_argument("--run-dir", required=True, type=Path)
    review.add_argument("--context-dir", required=True, type=Path)
    review.add_argument("--truth-dir", required=True, type=Path)
    review.add_argument("--memory-bank", required=True, type=Path)
    review.add_argument("--output-dir", type=Path)
    review.add_argument("--reviewer-mode", choices=["code_only", "hybrid_llm"], default="code_only")
    review.add_argument("--config", type=Path, help="Optional config used for LLM reviewer settings and memory caps.")

    args = parser.parse_args(argv)
    if args.command == "validate-config":
        config = load_config(args.config)
        print(json.dumps(config.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        config = load_config(args.config)
        run_dir = asyncio.run(run_society(config))
        print(f"wrote society run: {run_dir}")
        return 0
    if args.command == "review-run":
        config = load_config(args.config) if args.config else None
        result = review_run(
            run_dir=args.run_dir,
            context_dir=args.context_dir,
            truth_dir=args.truth_dir,
            memory_bank=args.memory_bank,
            output_dir=args.output_dir,
            reviewer_mode=args.reviewer_mode,
            llm_config=config.llm if config is not None else None,
            max_items_per_agent=config.memory_max_items_per_agent if config is not None else 5,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2
