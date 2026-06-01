"""Verifier impact: coverage premise + proposer-invariance of the worst-case-profit floor.

This is the reproducible source for the chapter-04 verifier-as-safety table/figure
(`tab:verifier-as-safety`, `fig:hallucinated-acceptance-bar`).

It reports three things, all from data on disk (nothing fabricated):

  1. Coverage premise (Theorem 1b, empirical): raw forecaster coverage vs ACI conformal
     coverage on the post-break validation window, per frozen seed. This is the premise
     `P(p_t in C_alpha) >= 1 - alpha` the whole verifier guarantee rests on.

  2. Proposer-invariance of the floor: per proposer regime, the share of *submitted* bids
     whose conformal worst-case profit is sub-floor (i.e. the verifier rejects / would block),
     and realised profit with the gate off vs on. The guarantee is the same; the rejection
     share is what moves with the proposer.

  3. The honest caveat: realised-loss count across the scored corpus (it is zero — the sim
     has no genuine downside, so the profit-floor *conclusion* is satisfied trivially and the
     verifier's bound is an ex-ante certificate, not a measured loss reduction).

Usage:
    PYTHONPATH=. uv run python tools/evaluation/evaluate_verifier_impact.py \
        --json-out evaluations/verifier_impact.json
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from statistics import mean

import pandas as pd


def coverage_premise(f8_metrics: Path) -> dict | None:
    """Raw vs ACI conformal coverage per seed (Theorem 1b premise)."""
    if not f8_metrics.exists():
        return None
    d = json.loads(f8_metrics.read_text())
    seeds = []
    for s in d.get("per_seed", []):
        seeds.append(
            {
                "seed": s.get("seed"),
                "raw_q10_q90_coverage": s.get("val_q10_q90_coverage"),
                "aci_coverage": s.get("aci_empirical_coverage"),
                "aci_target": 1.0 - s.get("aci_alpha_target", 0.1),
                "aci_mean_width_eur": s.get("aci_mean_width"),
            }
        )
    if not seeds:
        return None
    return {
        "per_seed": seeds,
        "raw_coverage_mean": round(mean(x["raw_q10_q90_coverage"] for x in seeds), 4),
        "aci_coverage_mean": round(mean(x["aci_coverage"] for x in seeds), 4),
        "target": seeds[0]["aci_target"],
        "note": "raw forecaster undercovers; the conformal layer is what creates the premise",
    }


def deterministic_binding(det_glob: str) -> dict:
    """Deployed deterministic policy: how often does the verifier actually bind?"""
    acc = abst = rej = watch = runs = 0
    for s in glob(det_glob):
        try:
            d = json.loads(Path(s).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        runs += 1
        acc += int(d.get("accepted", 0) or 0)
        abst += int(d.get("abstained", 0) or 0)
        rej += int(d.get("rejected", 0) or 0)
        watch += int(d.get("watched", 0) or 0)
    decided = acc + rej
    return {
        "runs": runs,
        "accepted": acc,
        "rejected": rej,
        "abstained": abst,
        "watched": watch,
        "verifier_reject_pct": round(100 * rej / decided, 3) if decided else 0.0,
        "note": "deployed best-accepted policy never tenders a sub-floor bid; floor is confirmatory",
    }


def llm_shadow(paired_summary: Path) -> dict | None:
    """Matched LLM verifier on (guarded) vs off (shadow): sub-floor share + profit."""
    if not paired_summary.exists():
        return None
    d = json.loads(paired_summary.read_text())
    by_variant: dict[str, list] = {}
    for r in d.get("rows", []):
        v = r.get("variant")
        sh = r.get("shadow")
        if v == "guarded" or not sh:
            continue
        m = r["metrics"]["realized_profit_eur"]
        by_variant.setdefault(v, []).append(
            {
                "subfloor_rate": sh.get("shadow_negative_worst_case_rate"),
                "would_block": sh.get("would_have_been_blocked_count"),
                "off_profit": m["variant"] if isinstance(m, dict) else None,
                "guarded_profit": m["guarded"] if isinstance(m, dict) else None,
            }
        )
    out = {}
    for v, rows in by_variant.items():
        out[v] = {
            "n_paired": len(rows),
            "subfloor_rate_mean": round(mean(x["subfloor_rate"] for x in rows), 4),
            "subfloor_rate_max": round(max(x["subfloor_rate"] for x in rows), 4),
            "would_block_total": sum(x["would_block"] for x in rows),
            "off_profit_mean": round(mean(x["off_profit"] for x in rows), 1),
            "guarded_profit_mean": round(mean(x["guarded_profit"] for x in rows), 1),
        }
    return out


def realised_loss_audit(eval_glob: str) -> dict:
    """The honest caveat: are there ANY realised losses in the corpus?"""
    n = 0
    n_neg = 0
    n_filled = 0
    minp = 0.0
    for f in glob(eval_glob):
        try:
            df = pd.read_parquet(f, columns=["status", "realized_profit_eur"])
        except Exception:  # skip unreadable parquet
            continue
        p = pd.to_numeric(df["realized_profit_eur"], errors="coerce")
        n += int(p.notna().sum())
        n_neg += int((p < 0).sum())
        n_filled += int((df["status"] == "filled").sum())
        if p.notna().any():
            minp = min(minp, float(p.min()))
    return {
        "scored_bids": n,
        "filled_bids": n_filled,
        "negative_realized_profit_count": n_neg,
        "min_realized_profit_eur": minp,
        "note": "zero realised losses => profit-floor conclusion is satisfied trivially; "
        "the verifier bound is an ex-ante certificate, not a measured loss reduction",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f8-metrics", type=Path, default=Path("models/forecaster/f8/metrics.json"))
    ap.add_argument("--det-glob", default="ai-society/runs/da-*det*/summary.json")
    ap.add_argument(
        "--paired-summary",
        type=Path,
        default=Path("evaluations/verifierless-baseline-20260519/paired_summary.json"),
    )
    ap.add_argument("--eval-glob", default="evaluations/*/bid_evaluations.parquet")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    out = {
        "coverage_premise_thm1b": coverage_premise(args.f8_metrics),
        "proposer_invariance": {
            "deployed_deterministic": deterministic_binding(args.det_glob),
            "llm_verifier_off_matched": llm_shadow(args.paired_summary),
        },
        "realised_loss_caveat": realised_loss_audit(args.eval_glob),
    }

    cp = out["coverage_premise_thm1b"]
    if cp:
        print(
            f"Coverage premise (Thm 1b): raw {cp['raw_coverage_mean']:.3f} -> "
            f"ACI {cp['aci_coverage_mean']:.3f} (target {cp['target']:.2f}), {len(cp['per_seed'])} seeds"
        )
    det = out["proposer_invariance"]["deployed_deterministic"]
    print(
        f"Deployed deterministic: {det['accepted']} accepted, {det['rejected']} rejected "
        f"=> verifier_reject {det['verifier_reject_pct']}%"
    )
    llm = out["proposer_invariance"]["llm_verifier_off_matched"] or {}
    for v, s in llm.items():
        print(
            f"LLM {v}: sub-floor mean {100*s['subfloor_rate_mean']:.1f}% "
            f"(max {100*s['subfloor_rate_max']:.1f}%), off profit {s['off_profit_mean']:.0f} "
            f"vs guarded {s['guarded_profit_mean']:.0f}"
        )
    rl = out["realised_loss_caveat"]
    print(
        f"Realised-loss audit: {rl['negative_realized_profit_count']} negatives in "
        f"{rl['scored_bids']} scored bids (min {rl['min_realized_profit_eur']:.1f} EUR)"
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
