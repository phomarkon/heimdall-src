"""Single-shot test-set evaluation (per docs/RESEARCH-PROPOSAL.md §5.7 + §10).

Critical: this script runs ONCE per final configuration on
``data/processed/dk1_panel_test.parquet``.  The leakage-detection hook from
``packages/data`` is wired in here; if a test-set evaluation has been run
before with the same `(model_name, seed, config_hash)` triple, we **refuse**
to run again unless `--allow-rerun` is passed (with reason).  This enforces
proposal §5.7's "test set is evaluated *once* per final configuration" rule.

Usage:
  PYTHONPATH=. python experiments/test_set_evaluation.py \\
      --models f0 f1_lgbm f2_blr f3 f3_ensemble f4_mc_dropout f7 f8 f9 \\
      --seeds 13 42 137 1729 31415 \\
      --output experiments/outputs/test_set_results.json

By default this is a *dry run* that lists what *would* be evaluated; pass
`--commit` to actually compute test-set metrics.  The dry-run guardrail is
intentional — accidental test-set leakage is the most expensive mistake on
the project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "models/forecaster"
TEST_PANEL = REPO_ROOT / "data/processed/dk1_panel_test.parquet"
LEDGER_PATH = REPO_ROOT / "experiments/outputs/test_set_ledger.json"
TEST_OUTPUT = REPO_ROOT / "experiments/outputs/test_set_results.json"

FROZEN_SEEDS = (13, 42, 137, 1729, 31415)


def _config_hash(model_name: str, seed: int, extras: dict) -> str:
    payload = json.dumps(
        {"model": model_name, "seed": seed, "extras": extras},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_ledger() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text())
    return {"runs": []}


def _save_ledger(ledger: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2))


def _pinball(y, q, level):
    err = y - q
    return float(np.mean(np.maximum(level * err, (level - 1.0) * err)))


def _evaluate_one(model_name: str, seed: int) -> dict:
    """Load the registered forecaster and run it on the test panel.

    Routes through ``heimdall_forecaster.inference.get_forecaster`` so
    every backend in the registry (f0, f7, f8, f10, f11, ...) is
    automatically supported — no per-model branch here.  Returns
    pinball loss at q10/q50/q90 and the q10-q90 marginal coverage.
    """
    import polars as pl

    from heimdall_forecaster.inference import get_forecaster

    seed_dir = MODEL_ROOT / model_name / f"seed-{seed}"
    metrics_path = seed_dir / "metrics.json"
    val_metrics_snapshot = (
        json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    )

    # Multivariate models (f8b/c/d) need the rich test panel + their
    # training feature set; everything else uses univariate imbalance prices.
    rich_models = {"f8b", "f8c", "f8d", "f8e", "f12", "f13"}
    if model_name in rich_models:
        from heimdall_forecaster.train.dataset import (
            F8B_FEATURES, F8C_FEATURES, F8D_FEATURES, F8E_FEATURES, F13_FEATURES,
        )
        feat_map = {"f8b": F8B_FEATURES, "f8c": F8C_FEATURES, "f8d": F8D_FEATURES,
                    "f8e": F8E_FEATURES, "f12": F8B_FEATURES, "f13": F13_FEATURES}
        feat_names = feat_map[model_name]
        rich_test = (
            REPO_ROOT / "data/processed/dk1_panel_rich_v2_test.parquet"
            if model_name == "f13"
            else REPO_ROOT / "data/processed/dk1_panel_rich_test.parquet"
        )
        df = pl.read_parquet(rich_test).sort("timestamp_utc")
        if model_name in {"f8c", "f8d"}:
            anom = pl.read_parquet(REPO_ROOT / "data/processed/anomaly_features_test.parquet")
            df = df.join(anom, on="timestamp_utc", how="left")
        missing = [c for c in feat_names if c not in df.columns]
        if missing:
            raise RuntimeError(f"{model_name}: missing test features {missing}")
        feat_arr = df.select(list(feat_names)).to_numpy().astype(float)
        # Fill NaNs forward then 0.
        feat_arr = np.nan_to_num(feat_arr, nan=0.0)
        series = df["imbalance_price_dkk_mwh_15min"].to_numpy().astype(float)
    else:
        df = pl.read_parquet(TEST_PANEL).sort("timestamp_utc")
        series = (
            df["imbalance_price_dkk_mwh_15min"].drop_nulls().to_numpy().astype(float)
        )
        feat_arr = None

    seq_len = 192
    horizon = 16
    n_windows = max(0, series.size - seq_len - horizon)
    if n_windows == 0:
        raise RuntimeError(f"test panel too short: only {series.size} rows")

    f = get_forecaster(model_name, seed)

    # Evaluate with a generous cap (10x the prior 1000) — covers ~104 days
    # of the year-long test set, enough for stratified hour/weekday metrics
    # without the all-day inference cost of the full 35k-window panel.
    n_eval = min(10000, n_windows)
    levels = (0.1, 0.5, 0.9)
    targets = np.empty((n_eval, horizon))
    preds = np.empty((n_eval, horizon, len(levels)))
    # Capture the issue-time timestamp of each window for stratified metrics.
    if "timestamp_utc" in df.columns:
        ts_all = df["timestamp_utc"].to_numpy()
        ts_issue = ts_all[seq_len : seq_len + n_eval]
    else:
        ts_issue = None
    for i in range(n_eval):
        target = series[i + seq_len : i + seq_len + horizon]
        targets[i] = target
        if feat_arr is not None:
            history = feat_arr[i : i + seq_len]
        else:
            history = list(series[i : i + seq_len])
        try:
            qfs = f.predict(history, horizon=horizon, levels=levels)
        except Exception as e:
            return {
                "model": model_name,
                "seed": seed,
                "val_metrics_snapshot": val_metrics_snapshot,
                "error": f"predict() failed at window {i}: {e!r}",
            }
        for h in range(horizon):
            preds[i, h, :] = qfs[h].values[: len(levels)]

    # Persist raw test preds + targets + timestamps for downstream metrics
    # re-runs without re-inference.
    test_preds_dir = REPO_ROOT / "experiments/outputs/test_preds" / model_name
    test_preds_dir.mkdir(parents=True, exist_ok=True)
    npz_payload = {"preds": preds.astype(np.float32),
                   "targets": targets.astype(np.float32)}
    if ts_issue is not None:
        npz_payload["timestamps"] = ts_issue.astype("datetime64[s]")
    np.savez(test_preds_dir / f"seed-{seed}.npz", **npz_payload)

    # Comprehensive metrics via heimdall_ml.eval.metrics.
    from heimdall_ml.eval import metrics as M
    comp = M.collect_all(preds, targets, levels, timestamps=ts_issue)

    per_q = {}
    for qi, q in enumerate(levels):
        err = targets - preds[..., qi]
        per_q[f"test_pinball_q{int(q * 100)}"] = float(
            np.mean(np.maximum(q * err, (q - 1.0) * err))
        )
    pinball_mean = float(np.mean(list(per_q.values())))
    sorted_p = np.sort(preds, axis=-1)
    coverage = float(np.mean((targets >= sorted_p[..., 0]) & (targets <= sorted_p[..., -1])))

    return {
        "model": model_name,
        "metrics": comp,
        "seed": seed,
        "val_metrics_snapshot": val_metrics_snapshot,
        "test_n_windows": n_eval,
        **per_q,
        "test_pinball_mean_dkk": pinball_mean,
        "test_q10_q90_coverage": coverage,
    }


@dataclass
class _RunSpec:
    model: str
    seed: int
    extras: dict = field(default_factory=dict)

    @property
    def cfg_hash(self) -> str:
        return _config_hash(self.model, self.seed, self.extras)


def _expand(models: Iterable[str], seeds_: Iterable[int]) -> list[_RunSpec]:
    return [_RunSpec(model=m, seed=s) for m in models for s in seeds_]


def _check_leakage(specs: list[_RunSpec], ledger: dict, allow_rerun: bool) -> list[_RunSpec]:
    seen = {entry["cfg_hash"] for entry in ledger.get("runs", [])}
    fresh: list[_RunSpec] = []
    blocked: list[_RunSpec] = []
    for spec in specs:
        if spec.cfg_hash in seen and not allow_rerun:
            blocked.append(spec)
        else:
            fresh.append(spec)
    if blocked:
        msg = "; ".join(f"{s.model}-seed{s.seed}({s.cfg_hash})" for s in blocked)
        print(
            "REFUSED: the following test-set evaluations have already been "
            "executed and would constitute a re-evaluation that proposal §5.7 "
            "forbids: " + msg + ". Pass --allow-rerun with --reason to override.",
            file=sys.stderr,
        )
        if not allow_rerun:
            sys.exit(2)
    return fresh


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(FROZEN_SEEDS))
    p.add_argument("--commit", action="store_true",
                   help="actually run; default is a dry run that prints the plan")
    p.add_argument("--allow-rerun", action="store_true")
    p.add_argument("--reason", type=str, default="")
    p.add_argument("--output", type=Path, default=TEST_OUTPUT)
    args = p.parse_args()

    if not TEST_PANEL.exists():
        print(f"FATAL: test panel not found at {TEST_PANEL}", file=sys.stderr)
        return 1

    specs = _expand(args.models, args.seeds)
    ledger = _load_ledger()
    fresh = _check_leakage(specs, ledger, args.allow_rerun)

    print(f"Plan: {len(fresh)} fresh test-set evaluations of "
          f"{len(set(s.model for s in fresh))} models × {len(args.seeds)} seeds.")
    for s in fresh:
        print(f"  - {s.model:20s} seed={s.seed} cfg={s.cfg_hash}")
    if not args.commit:
        print("\nDRY RUN — pass --commit to actually evaluate.")
        return 0

    if args.allow_rerun and not args.reason:
        print("FATAL: --allow-rerun requires --reason", file=sys.stderr)
        return 2

    started = time.time()
    results: list[dict] = []
    for s in fresh:
        try:
            r = _evaluate_one(s.model, s.seed)
            r["cfg_hash"] = s.cfg_hash
            results.append(r)
            ledger.setdefault("runs", []).append(
                {
                    "cfg_hash": s.cfg_hash,
                    "model": s.model,
                    "seed": s.seed,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "reason": args.reason or "single-shot",
                }
            )
        except Exception as e:
            print(f"FAILED {s.model} seed={s.seed}: {e}", file=sys.stderr)
            results.append({"model": s.model, "seed": s.seed, "error": str(e)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Append-merge into existing results file so multi-call runs don't drop
    # prior models. Key on (model, seed) with the latest run winning.
    existing = {}
    if args.output.exists():
        try:
            prev = json.loads(args.output.read_text())
            for r in prev.get("results", []):
                existing[(r.get("model"), r.get("seed"))] = r
        except Exception:
            existing = {}
    for r in results:
        existing[(r.get("model"), r.get("seed"))] = r
    args.output.write_text(json.dumps({"results": list(existing.values())}, indent=2))
    _save_ledger(ledger)
    print(f"Done — {len(results)} evaluations in {time.time() - started:.1f}s.")
    print(f"Wrote {args.output}")
    print(f"Wrote ledger {LEDGER_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
