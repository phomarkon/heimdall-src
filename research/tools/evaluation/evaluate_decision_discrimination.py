"""D4-v2 decision DISCRIMINATION — does the LLM's self-reported signal carry
information about the realized outcome that the deterministic core does not emit?

D4-v1 (evaluate_decision_calibration.py) reported only ECE (calibration *level*),
which is trivially fixable by a post-hoc isotonic map (0.85 -> base rate) and proves
nothing about value. The value question is DISCRIMINATION / RESOLUTION: does a higher
LLM confidence / lower risk_label actually rank-order which bids fill, win, or pick the
right side? A deterministic limit order emits no such forward-looking self-assessment,
so if the LLM's labels discriminate (AUC >> 0.5) while the deterministic control's
templated labels do not (AUC ~ 0.5), that is value the deterministic core cannot provide.

Honest design (per feedback-value-metrics-need-clean-controls):
  * Deterministic-chooser runs are the CONTROL arm. Their templated confidence MUST
    score ~0.5 AUC / ~0 resolution. If a control discriminates, the metric is leaking.
  * Outcomes are the realized truth join (_score_bids over activation_truth), not a proxy.
  * `profitable` is reported but flagged degenerate (base rate ~0.07, no downside in sim);
    `filled` and `clean_side` carry the real signal.

AUC is the Mann-Whitney rank statistic (P[signal higher on positive than on negative]).
Brier is decomposed into reliability - resolution + uncertainty (Murphy 1973): resolution
is the part that matters (how much the signal separates outcome base rates).

Usage:
    PYTHONPATH=. uv run python tools/evaluation/evaluate_decision_discrimination.py \
        --glob 'ai-society/runs/**/' \
        --truth-dir data/cache/evaluation_truth/april_2026 \
        --json-out evaluations/decision_discrimination.json
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluation.evaluate_society_run import _load_traces, _load_truth, _score_bids

SUBMITTED = {"filled", "partially_filled", "wrong_side", "price_not_crossed"}
FILLED = {"filled", "partially_filled"}
RISK_ORD = {"low": 0.0, "medium": 1.0, "high": 2.0}
OPP_ORD = {"none": 0.0, "weak": 1.0, "actionable": 2.0}
DECISION_COLS = ["run_id", "agent_id", "step", "confidence", "risk_label",
                 "uncertainty_label", "priority_score", "opportunity_label", "action"]


def _auc(signal: np.ndarray, label: np.ndarray) -> tuple[float, int, int]:
    """Mann-Whitney AUC = P(signal_pos > signal_neg) + 0.5 P(tie). Returns (auc, n_pos, n_neg)."""
    mask = ~np.isnan(signal)
    signal, label = signal[mask], label[mask]
    pos = signal[label == 1]
    neg = signal[label == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    concat = np.concatenate([pos, neg])[order]
    # average ranks for ties
    ranks_sorted = np.arange(1, len(concat) + 1, dtype=float)
    i = 0
    while i < len(concat):
        j = i
        while j + 1 < len(concat) and concat[j + 1] == concat[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    inv = np.empty(len(order), dtype=int)
    inv[order] = np.arange(len(order))
    rank_pos = ranks_sorted[inv[:n_pos]]
    auc = (rank_pos.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg


def _brier_decomp(conf: np.ndarray, outcome: np.ndarray, bins: int = 10) -> dict:
    mask = ~np.isnan(conf)
    conf, outcome = conf[mask], outcome[mask]
    n = len(conf)
    if n == 0:
        return {"brier": float("nan"), "reliability": float("nan"),
                "resolution": float("nan"), "uncertainty": float("nan")}
    base = outcome.mean()
    edges = np.linspace(0.0, 1.0, bins + 1)
    rel = res = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        nk = m.sum()
        if nk == 0:
            continue
        conf_k = conf[m].mean()
        out_k = outcome[m].mean()
        rel += nk * (conf_k - out_k) ** 2
        res += nk * (out_k - base) ** 2
    rel /= n
    res /= n
    unc = base * (1 - base)
    return {"brier": float(((conf - outcome) ** 2).mean()), "reliability": float(rel),
            "resolution": float(res), "uncertainty": float(unc),
            "brier_skill_vs_base": float(1 - ((conf - outcome) ** 2).mean() / unc) if unc > 0 else float("nan")}


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), len(x)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), len(x)
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


def _arm_of(summary: dict) -> str:
    """LLM-decision run vs deterministic control, from summary.json classification."""
    if not summary.get("llm_enabled", False):
        return "det"
    chooser = str(summary.get("chooser_mode", ""))
    abl = str(summary.get("ablation_strategy", ""))
    # deterministic chooser even with llm_enabled (e.g. rationale-only) -> control
    if chooser.startswith("deterministic") or chooser in {"best_accepted", ""}:
        return "det"
    return f"llm:{abl or chooser}"


def _bucket(arm: str) -> str:
    return "det" if arm == "det" else "llm"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--truth-dir", type=Path, default=Path("data/cache/evaluation_truth/april_2026"))
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--by-arm", action="store_true", help="also break LLM down by ablation strategy")
    args = ap.parse_args()

    truth = _load_truth(args.truth_dir / "activation_truth.parquet")
    run_dirs: list[Path] = [Path(r) for r in args.runs]
    for g in args.glob:
        run_dirs += [Path(p) for p in glob(g, recursive=True)]
    run_dirs = sorted({d for d in run_dirs if (d / "traces.jsonl").exists() and (d / "summary.json").exists()})
    if not run_dirs:
        raise SystemExit("no runs with traces.jsonl + summary.json")

    frames: dict[str, list[pd.DataFrame]] = {}
    n_runs = {"det": 0, "llm": 0}
    arms_seen: dict[str, int] = {}
    for d in run_dirs:
        try:
            summary = json.loads((d / "summary.json").read_text())
        except json.JSONDecodeError:
            continue
        arm = _arm_of(summary)
        bucket = _bucket(arm)
        traces = _load_traces(d / "traces.jsonl")
        if traces.empty:
            continue
        bids = _score_bids(traces, truth)
        if bids.empty:
            continue
        have = [c for c in DECISION_COLS if c in traces.columns]
        dec = traces[have].drop_duplicates(["run_id", "agent_id", "step"])
        merged = bids.merge(dec, on=["run_id", "agent_id", "step"], how="left", suffixes=("", "_dec"))
        merged["__arm"] = arm
        frames.setdefault(bucket, []).append(merged)
        if args.by_arm and bucket == "llm":
            frames.setdefault(arm, []).append(merged)
        n_runs[bucket] += 1
        arms_seen[arm] = arms_seen.get(arm, 0) + 1

    out: dict = {"n_runs": n_runs, "arms": arms_seen, "groups": {}}
    print(f"\nD4-v2 decision discrimination — {len(run_dirs)} runs "
          f"(det={n_runs['det']}, llm={n_runs['llm']})\n")
    print("Reads: AUC>0.5 = signal ranks the outcome; det control should sit at ~0.50.")
    print("Signals: confidence (higher=better expected), risk/uncertainty (encoded so "
          "higher=worse), priority_score, opportunity (higher=better).\n")

    order = ["det", "llm"] + ([a for a in sorted(arms_seen) if a != "det"] if args.by_arm else [])
    for group in order:
        if group not in frames:
            continue
        df = pd.concat(frames[group], ignore_index=True)
        sub = df[df["status"].isin(SUBMITTED)].copy()
        n = len(sub)
        if n == 0:
            print(f"### {group}: no submitted bids\n")
            continue
        conf = pd.to_numeric(sub.get("confidence"), errors="coerce").to_numpy()
        prio = pd.to_numeric(sub.get("priority_score"), errors="coerce").to_numpy()
        risk = sub.get("risk_label").map(RISK_ORD).to_numpy() if "risk_label" in sub else np.full(n, np.nan)
        unc = sub.get("uncertainty_label").map(RISK_ORD).to_numpy() if "uncertainty_label" in sub else np.full(n, np.nan)
        opp = sub.get("opportunity_label").map(OPP_ORD).to_numpy() if "opportunity_label" in sub else np.full(n, np.nan)

        filled = sub["status"].isin(FILLED).astype(float).to_numpy()
        profitable = (pd.to_numeric(sub["realized_profit_eur"], errors="coerce") > 0).astype(float).to_numpy()
        clean_side = (sub["status"] != "wrong_side").astype(float).to_numpy()
        ppm = pd.to_numeric(sub["profit_per_mwh"], errors="coerce").to_numpy()

        base_fill = filled.mean()
        auc_cf, npf, nnf = _auc(conf, filled)
        auc_cp, _, _ = _auc(conf, profitable)
        auc_cs, _, _ = _auc(conf, clean_side)
        auc_rf, _, _ = _auc(-risk, filled)      # low risk -> fill
        auc_rs, _, _ = _auc(-risk, clean_side)   # low risk -> right side
        auc_pf, _, _ = _auc(prio, filled)
        auc_of, _, _ = _auc(opp, filled)
        bd_fill = _brier_decomp(conf, filled)
        sp_ppm, n_ppm = _spearman(conf, ppm)

        print(f"### {group}  (submitted={n}, fill_rate={base_fill:.2f}, "
              f"P(profitable)={profitable.mean():.3f}, conf std={np.nanstd(conf):.3f})")
        print(f"   AUC confidence -> filled     : {auc_cf:.3f}   (n+={npf}, n-={nnf})  [main]")
        print(f"   AUC confidence -> clean_side : {auc_cs:.3f}   (right-side discrimination)")
        print(f"   AUC confidence -> profitable : {auc_cp:.3f}   (degenerate base rate, weak)")
        print(f"   AUC low-risk    -> filled    : {auc_rf:.3f}")
        print(f"   AUC low-risk    -> clean_side: {auc_rs:.3f}")
        print(f"   AUC priority    -> filled    : {auc_pf:.3f}")
        print(f"   AUC opportunity -> filled    : {auc_of:.3f}")
        print(f"   Brier(conf,filled)={bd_fill['brier']:.3f}  resolution={bd_fill['resolution']:.4f} "
              f"reliability={bd_fill['reliability']:.4f}  skill_vs_base={bd_fill['brier_skill_vs_base']:.3f}")
        print(f"   Spearman conf vs profit/MWh (filled only): {sp_ppm:.3f} (n={n_ppm})")
        print()

        out["groups"][group] = {
            "submitted": int(n), "fill_rate": float(base_fill),
            "P_profitable": float(profitable.mean()), "conf_std": float(np.nanstd(conf)),
            "auc_conf_filled": auc_cf, "auc_conf_clean_side": auc_cs, "auc_conf_profitable": auc_cp,
            "auc_lowrisk_filled": auc_rf, "auc_lowrisk_clean_side": auc_rs,
            "auc_priority_filled": auc_pf, "auc_opportunity_filled": auc_of,
            "brier_conf_filled": bd_fill, "spearman_conf_ppm": sp_ppm,
        }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
