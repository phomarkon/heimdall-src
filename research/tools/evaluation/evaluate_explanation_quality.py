"""Explanation completeness + selectivity vs the trace ground truth (D3 extension).

D3 (`evaluate_rationale_faithfulness.py`) answers "is the rationale TRUE?" (faithfulness:
no confabulated evidence). It does NOT answer "is the rationale COMPLETE and decision-relevant?"
A constant string can be perfectly faithful and explain nothing — the deterministic baseline
emits the *same* sentence for every bid ("deterministic_best_accepted selected the highest-ranked
exact simulator-accepted candidate"), naming zero market drivers.

This tool measures the missing axis, with the same anti-confabulation discipline:

  founded coverage   — of the decision drivers actually present in this tick's trace
                       (side edge, profitability, feasibility, regime, uncertainty, constraint),
                       what fraction does the rationale name AND is the evidence really there?
                       A driver named without supporting trace evidence earns NO coverage credit
                       and is counted as `unfounded` (this is the D3 link: completeness is only
                       credited when faithful).
  selectivity        — on the ticks that carry a *distinctive* driver (the bid goes against the
                       activation hint; uncertainty is high / side is ambiguous), does the
                       rationale surface that specific tension? This is the contrastive,
                       per-tick-adaptive part a template cannot fake.
  autonomous split   — records with >=1 LLM-requested simulate call, separating an LLM that
                       *reasons* from one that *transcribes* a code-ranked menu (selector mode).

Controls (must score the trivial value before the metric is trusted):
  det_rich       constant procedural string -> coverage ~ acceptance-only, selectivity 0.
  selector_rich  rich text but autonomous ~ 0% (narrates the menu).
  grounded LLM   high founded coverage, high selectivity, autonomous ~ 1.0.
  ungrounded LLM raw coverage may look high but founded coverage drops + unfounded rises
                 (it confabulates drivers it never gathered).

Usage:
    python tools/evaluation/evaluate_explanation_quality.py <run_dir> [more...]
    python tools/evaluation/evaluate_explanation_quality.py --glob 'ai-society/runs/d3-*/*'
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

NUM_TOL = 1.0

# --- driver detectors: a rationale "names" a driver if its surface form appears ----------------
RE_SIDE = re.compile(r"\b(up-?edge|down-?edge|up-?bid|down-?bid|up-?side|down-?side|"
                     r"upward|downward|above .{0,12}spot|below .{0,12}spot|edge .{0,12}spot|"
                     r"price edge|spot price|capture .{0,12}edge)\b", re.IGNORECASE)
RE_PROFIT = re.compile(r"\b(profit|worst[- ]?case|expected[- ]?profit|margin|EUR\b)", re.IGNORECASE)
RE_ACCEPT = re.compile(r"\b(accepted|simulator|feasible|clearable|passes|candidate|probe)\b", re.IGNORECASE)
RE_REGIME = re.compile(r"\b(volatile|quiet|normal|high[- ]?activation|negative[- ]?price)\b", re.IGNORECASE)
RE_UNC = re.compile(r"\b(uncertain|uncertainty|ambiguity|ambiguous|interval width|forecast width)\b", re.IGNORECASE)
RE_CONSTRAINT = re.compile(r"\b(export pressure|import pressure|congestion|grid constraint|cnec|"
                           r"outage|capacity|soc|state[- ]?of[- ]?charge|ramp|border|flow)\b", re.IGNORECASE)
# contrastive side-justification: either name the opposing signal/tension, OR contrastively
# compare the two sides' edges ("up-edge exceeds down-edge" justifies WHY this side, not the other).
RE_TENSION = re.compile(r"\b(asymmetric|against|despite|contrar|tension|opposing|conflict|"
                        r"direction hint|activation .{0,12}hint|however|while|whereas|even though|"
                        r"exceeds|dominates|outweighs|stronger than|deficit|vs\.?|versus)\b",
                        re.IGNORECASE)
# regime words for correctness check
REGIME_WORDS = ("volatile", "quiet", "normal", "high_activation", "high activation",
                "negative_price", "negative price")
RE_EUR = re.compile(r"(-?\d+(?:\.\d+)?)\s*EUR", re.IGNORECASE)

DRIVERS = ("side", "profit", "accept", "regime", "uncertainty", "constraint")


@dataclass
class ArmTally:
    n_records: int = 0
    n_bids: int = 0
    # coverage: per-driver applicable/named/founded counts (summed over bids)
    applicable: dict = field(default_factory=lambda: {d: 0 for d in DRIVERS})
    named: dict = field(default_factory=lambda: {d: 0 for d in DRIVERS})
    founded: dict = field(default_factory=lambda: {d: 0 for d in DRIVERS})
    # per-bid coverage fractions (for the mean)
    founded_cov_sum: float = 0.0
    raw_cov_sum: float = 0.0
    cov_bids: int = 0
    unfounded_driver_mentions: int = 0  # driver named but evidence absent (confabulated)
    # selectivity (distinctive cases)
    tension_cases: int = 0
    tension_named: int = 0
    highunc_cases: int = 0
    highunc_named: int = 0
    autonomous_records: int = 0
    examples: list = field(default_factory=list)


def _trace_ground_truth(tool_calls: list[dict]) -> dict:
    """Extract the decision drivers actually present in this tick's tool results."""
    g = {
        "regime_label": None, "uncertainty_label": None, "side_ambiguity": None,
        "up_edge": None, "down_edge": None, "activation_hint": None,
        "pressure_label": None, "grid_rows": 0, "outage_impact": None,
        "sim_profit": set(), "accepted": False, "edge_present": False,
        "has_llm_sim": False, "constraint_present": False,
    }
    for tc in tool_calls:
        name, res = tc.get("name", ""), tc.get("result")
        if not isinstance(res, dict):
            continue
        if name == "get_market_regime_context":
            g["regime_label"] = res.get("regime_label")
            sig = res.get("signals") or {}
            if "up_edge_lower_minus_spot_eur_mwh" in sig:
                g["up_edge"] = sig.get("up_edge_lower_minus_spot_eur_mwh")
                g["down_edge"] = sig.get("down_edge_spot_minus_upper_eur_mwh")
                g["edge_present"] = True
            g["activation_hint"] = g["activation_hint"] or sig.get("activation_direction_hint")
        elif name == "get_uncertainty_digest":
            g["uncertainty_label"] = res.get("uncertainty_label")
            g["side_ambiguity"] = res.get("side_ambiguity")
            sig = res.get("signals") or {}
            if "up_edge_lower_minus_spot_eur_mwh" in sig:
                g["edge_present"] = True
        elif name in ("get_activation_context", "get_opportunity_context"):
            g["activation_hint"] = g["activation_hint"] or res.get("direction_hint")
        elif name == "get_border_pressure":
            g["pressure_label"] = res.get("pressure_label")
            if res.get("pressure_label") and res["pressure_label"] not in ("neutral", "balanced", None):
                g["constraint_present"] = True
        elif name == "get_grid_constraints":
            g["grid_rows"] = res.get("row_count") or 0
            if g["grid_rows"]:
                g["constraint_present"] = True
        elif name in ("get_outage_impact", "get_outages"):
            lbl = res.get("impact_label")
            if lbl and lbl != "none":
                g["outage_impact"] = lbl
                g["constraint_present"] = True
        elif name.startswith("simulate"):
            for k in ("worst_case_profit_eur", "expected_profit_eur"):
                v = res.get(k)
                if isinstance(v, (int, float)):
                    g["sim_profit"].add(round(float(v), 1))
            if res.get("accepted"):
                g["accepted"] = True
            args = tc.get("arguments") or {}
            # SoC / capacity constraint surfaced by the sim itself
            if any(kk in res for kk in ("soc_mwh", "available_quantity_mwh")):
                g["constraint_present"] = True
            if tc.get("provenance") == "llm_requested":
                g["has_llm_sim"] = True
    return g


def score_record(rec: dict, t: ArmTally) -> None:
    decision = rec.get("decision") or {}
    text = (decision.get("rationale") or rec.get("rationale") or "").strip()
    if not text:
        return
    t.n_records += 1
    if decision.get("action") != "bid":
        # only bids carry a full decision to explain; still count autonomy
        g0 = _trace_ground_truth(rec.get("tool_calls") or [])
        if g0["has_llm_sim"]:
            t.autonomous_records += 1
        return
    t.n_bids += 1
    g = _trace_ground_truth(rec.get("tool_calls") or [])
    if g["has_llm_sim"]:
        t.autonomous_records += 1
    side = decision.get("side")

    # ---- applicability of each driver in THIS tick ----
    appl = {
        "side": g["edge_present"],                       # there is a side edge to justify
        "profit": len(g["sim_profit"]) > 0,              # a sim returned profit
        "accept": g["accepted"],                         # an accepted sim exists
        "regime": g["regime_label"] is not None,
        "uncertainty": g["uncertainty_label"] is not None,
        "constraint": g["constraint_present"],           # a real constraint/pressure exists
    }
    # ---- named (surface form present) ----
    named = {
        "side": bool(RE_SIDE.search(text)),
        "profit": bool(RE_PROFIT.search(text)),
        "accept": bool(RE_ACCEPT.search(text)),
        "regime": bool(RE_REGIME.search(text)),
        "uncertainty": bool(RE_UNC.search(text)),
        "constraint": bool(RE_CONSTRAINT.search(text)),
    }
    # ---- founded: named AND evidence really present AND (for verifiable ones) correct ----
    founded = {}
    for d in DRIVERS:
        if not named[d]:
            founded[d] = False
            continue
        if d == "regime":
            # correctness: claimed regime word matches the trace label
            claimed = [w for w in REGIME_WORDS if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)]
            lbl = (g["regime_label"] or "").replace("_watch", "").replace("_", " ")
            founded[d] = bool(claimed) and any(c.replace("_", " ") in (lbl + " " + (g["regime_label"] or "")) for c in claimed)
        elif d == "constraint":
            # founded only if a real constraint/pressure/outage was actually retrieved
            founded[d] = g["constraint_present"]
        elif d == "accept":
            founded[d] = g["accepted"]
        elif d == "profit":
            founded[d] = len(g["sim_profit"]) > 0
        elif d == "side":
            founded[d] = g["edge_present"]
        elif d == "uncertainty":
            founded[d] = g["uncertainty_label"] is not None
        # named-but-not-founded == confabulated mention
        if named[d] and not founded[d]:
            t.unfounded_driver_mentions += 1

    n_appl = sum(appl.values())
    if n_appl:
        founded_hit = sum(1 for d in DRIVERS if appl[d] and founded[d])
        raw_hit = sum(1 for d in DRIVERS if appl[d] and named[d])
        t.founded_cov_sum += founded_hit / n_appl
        t.raw_cov_sum += raw_hit / n_appl
        t.cov_bids += 1
    for d in DRIVERS:
        t.applicable[d] += int(appl[d])
        t.named[d] += int(named[d] and appl[d])
        t.founded[d] += int(founded[d] and appl[d])

    # ---- selectivity: distinctive cases ----
    # (A) bid goes against the activation hint -> a complete explanation must justify it
    if side and g["activation_hint"] in ("up", "down") and side != g["activation_hint"]:
        t.tension_cases += 1
        # credit if it names the hint/opposing direction OR a tension word
        if RE_TENSION.search(text) or re.search(rf"\b{g['activation_hint']}\b", text, re.IGNORECASE):
            t.tension_named += 1
        elif len(t.examples) < 6:
            t.examples.append(f"[{rec.get('agent_id')} s{rec.get('step')}] bid {side} vs hint {g['activation_hint']}, "
                              f"no tension cited: {text[:110]}")
    # (B) high uncertainty / side ambiguity -> must flag it
    if g["uncertainty_label"] == "high" or g["side_ambiguity"] is True:
        t.highunc_cases += 1
        if RE_UNC.search(text) or "ambig" in text.lower():
            t.highunc_named += 1


def score_run(run_dir: Path) -> ArmTally:
    t = ArmTally()
    p = run_dir / "traces.jsonl"
    if not p.exists():
        return t
    for line in p.read_text().splitlines():
        if line.strip():
            score_record(json.loads(line), t)
    return t


def arm_name(run_dir: Path) -> str:
    parts = run_dir.name.split("-")
    # lvar-<arm>-... / d3-<arm>-... / lva-<arm>-...
    return parts[1] if len(parts) > 1 else run_dir.name


def _merge(agg: ArmTally, t: ArmTally) -> None:
    for f in ("n_records", "n_bids", "founded_cov_sum", "raw_cov_sum", "cov_bids",
              "unfounded_driver_mentions", "tension_cases", "tension_named",
              "highunc_cases", "highunc_named", "autonomous_records"):
        setattr(agg, f, getattr(agg, f) + getattr(t, f))
    for d in DRIVERS:
        agg.applicable[d] += t.applicable[d]
        agg.named[d] += t.named[d]
        agg.founded[d] += t.founded[d]
    agg.examples = (agg.examples + t.examples)[:6]


def _pct(num: int, den: int) -> str:
    return f"{100*num/den:.0f}%" if den else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--per-driver", action="store_true", help="print per-driver founded-coverage")
    args = ap.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    if args.glob:
        run_dirs += [Path(p) for p in glob(args.glob)]
    run_dirs = [d for d in run_dirs if (d / "traces.jsonl").exists()]
    if not run_dirs:
        raise SystemExit("no run dirs with traces.jsonl")

    arms: dict[str, ArmTally] = {}
    for d in sorted(run_dirs):
        _merge(arms.setdefault(arm_name(d), ArmTally()), score_run(d))

    print(f"\nExplanation completeness + selectivity — {len(run_dirs)} runs, {len(arms)} arms\n")
    hdr = (f"{'arm':<16}{'bids':>5}{'found-cov':>10}{'raw-cov':>9}{'unfounded':>10}"
           f"{'tension':>9}{'high-unc':>9}{'autonom':>9}")
    print(hdr); print("-" * len(hdr))
    out = {}
    for arm in sorted(arms):
        t = arms[arm]
        fcov = t.founded_cov_sum / t.cov_bids if t.cov_bids else 0.0
        rcov = t.raw_cov_sum / t.cov_bids if t.cov_bids else 0.0
        print(f"{arm:<16}{t.n_bids:>5}{fcov:>10.2f}{rcov:>9.2f}"
              f"{t.unfounded_driver_mentions:>10}"
              f"{_pct(t.tension_named, t.tension_cases):>9}"
              f"{_pct(t.highunc_named, t.highunc_cases):>9}"
              f"{_pct(t.autonomous_records, t.n_records):>9}")
        out[arm] = {
            "n_records": t.n_records, "n_bids": t.n_bids,
            "founded_coverage": round(fcov, 3), "raw_coverage": round(rcov, 3),
            "unfounded_driver_mentions": t.unfounded_driver_mentions,
            "tension_cases": t.tension_cases, "tension_named": t.tension_named,
            "highunc_cases": t.highunc_cases, "highunc_named": t.highunc_named,
            "autonomous_pct": round(100 * t.autonomous_records / t.n_records, 1) if t.n_records else 0.0,
            "per_driver_founded": {d: [t.founded[d], t.applicable[d]] for d in DRIVERS},
        }
    print("\nLegend: found-cov=mean fraction of trace-present drivers the rationale names AND that "
          "are really in the trace (confabulated names get no credit); raw-cov=names regardless of "
          "evidence; unfounded=driver mentions with no supporting tool evidence (confabulation, ties "
          "to D3); tension=bids going against the activation hint that cite the tension; "
          "high-unc=high-uncertainty/ambiguous ticks that flag it; autonom=records with an "
          "LLM-requested simulate call (reasons vs transcribes a menu).\n")

    if args.per_driver:
        print(f"{'driver':<14}" + "".join(f"{a[:14]:>15}" for a in sorted(arms)))
        for dr in DRIVERS:
            row = f"{dr:<14}"
            for a in sorted(arms):
                t = arms[a]
                row += f"{_pct(t.founded[dr], t.applicable[dr]):>15}"
            print(row)
        print()

    for arm in sorted(arms):
        if arms[arm].examples:
            print(f"### {arm} — distinctive cases the rationale missed")
            for e in arms[arm].examples[:4]:
                print("  MISS:", e)
            print()

    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
