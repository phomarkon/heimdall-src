"""Rationale faithfulness vs the trace ground truth (D3).

Replaces the syntactic rationale rubric's blind spot: instead of rewarding text for
*containing* numbers and tool names, this checks whether the factual claims a rationale
makes are TRUE against the same agent-tick's tool-call results. A claim is only verifiable
against evidence the agent actually retrieved, so we also flag claims about evidence the
agent never gathered ("unfounded" — e.g. "no outages detected" with no get_outages call).

Honest scope: this measures faithfulness to the trace, not decision quality. A perfectly
faithful rationale that merely transcribes a code-ranked menu (selector mode) adds no alpha.
The transcription vs autonomous split (by tool provenance) makes that distinction explicit.

Usage:
    python tools/evaluation/evaluate_rationale_faithfulness.py ai-society/runs/<matrix>/<run> [...more runs]
    python tools/evaluation/evaluate_rationale_faithfulness.py --glob 'ai-society/runs/llm-value-allarch*/*'
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

NUM_TOL = 1.0  # EUR tolerance for matching a quoted number to a trace value
REGIME_WORDS = ("volatile", "quiet", "normal", "negative_price", "negative price", "high_activation", "high activation")
# "X EUR/MWh" (edge/price) vs "X EUR" (profit). Order matters: match /MWh first.
RE_EUR_MWH = re.compile(r"(-?\d+(?:\.\d+)?)\s*EUR\s*/\s*MWh", re.IGNORECASE)
RE_EUR = re.compile(r"(-?\d+(?:\.\d+)?)\s*EUR(?!\s*/)", re.IGNORECASE)
OUTAGE_RE = re.compile(r"outage", re.IGNORECASE)
GRID_RE = re.compile(r"grid constraint|cnec|congestion", re.IGNORECASE)
# Negation tokens that, if they appear in the ~15 chars before an "accept", flip the meaning
# ("no accepted candidate", "not accepted", "n't accepted"). Local window only — a "rejected"
# earlier in a long sentence does not suppress a genuine later affirmation.
_NEG = ("no ", "not ", "n't", "never", "without")
_RE_ACCEPT = re.compile(r"accept", re.IGNORECASE)


def _affirms_acceptance(text: str) -> bool:
    for m in _RE_ACCEPT.finditer(text):
        ctx = text[max(0, m.start() - 15):m.start()].lower()
        if any(n in ctx for n in _NEG):
            continue
        return True
    return False


def _bid_matches_accepted(decision: dict, accepted_sims: list[dict]) -> bool:
    side = decision.get("side")
    qty = decision.get("quantity_mwh")
    price = decision.get("limit_price_eur_mwh")
    for s in accepted_sims:
        if s.get("side") != side:
            continue
        if qty is not None and s.get("quantity_mwh") is not None and abs(float(s["quantity_mwh"]) - float(qty)) > 0.01:
            continue
        if price is not None and s.get("limit_price_eur_mwh") is not None and abs(float(s["limit_price_eur_mwh"]) - float(price)) > NUM_TOL:
            continue
        return True
    return False


@dataclass
class ArmTally:
    n_records: int = 0
    n_bids: int = 0
    # numeric grounding
    nums_total: int = 0
    nums_grounded: int = 0
    # acceptance claims
    accept_claims: int = 0
    accept_supported: int = 0
    accept_contradicted: int = 0  # hallucinated acceptance
    # regime claims
    regime_claims: int = 0
    regime_supported: int = 0
    regime_contradicted: int = 0
    regime_unfounded: int = 0  # regime word but no regime tool retrieved
    # evidence-existence claims (outage/grid): did the agent actually gather it?
    evidence_claims: int = 0
    evidence_unfounded: int = 0  # claims about evidence never retrieved
    # provenance split
    autonomous_sim_records: int = 0  # >=1 llm_requested simulate call
    examples_contradiction: list[str] = field(default_factory=list)
    examples_unfounded: list[str] = field(default_factory=list)


def _trace_numeric_bank(tool_calls: list[dict]) -> tuple[set[float], set[float], dict]:
    """Return (profit_values, edge_values, context) drawn from tool results."""
    profit: set[float] = set()
    edge: set[float] = set()
    ctx = {"regime_label": None, "outage_called": False, "grid_called": False,
           "outage_nonempty": None, "accepted_sims": [], "has_llm_sim": False}
    for tc in tool_calls:
        name = tc.get("name", "")
        res = tc.get("result")
        if not isinstance(res, dict):
            continue
        if name.startswith("simulate"):
            for k in ("worst_case_profit_eur", "expected_profit_eur"):
                v = res.get(k)
                if isinstance(v, (int, float)):
                    profit.add(round(float(v), 1))
            if res.get("accepted"):
                ctx["accepted_sims"].append({
                    "side": (tc.get("arguments") or {}).get("side"),
                    "quantity_mwh": (tc.get("arguments") or {}).get("quantity_mwh"),
                    "limit_price_eur_mwh": (tc.get("arguments") or {}).get("limit_price_eur_mwh"),
                })
            if tc.get("provenance") == "llm_requested":
                ctx["has_llm_sim"] = True
            fi = res.get("forecast_interval_eur_mwh")
            if isinstance(fi, list):
                edge.update(round(float(x), 1) for x in fi if isinstance(x, (int, float)))
        elif name == "get_market_regime_context":
            ctx["regime_label"] = res.get("regime_label")
        elif name in ("get_activation_context", "get_opportunity_context", "get_market_regime_context", "get_uncertainty_digest"):
            sig = res.get("signals")
            if isinstance(sig, dict):
                edge.update(round(float(v), 1) for v in sig.values() if isinstance(v, (int, float)))
        elif name in ("get_outages", "get_outage_impact"):
            ctx["outage_called"] = True
            r = res.get("outages") if isinstance(res.get("outages"), list) else res.get("events")
            if isinstance(r, list):
                ctx["outage_nonempty"] = len(r) > 0
        elif name in ("get_grid_constraints",):
            ctx["grid_called"] = True
    return profit, edge, ctx


def _grounded(value: float, bank: set[float]) -> bool:
    return any(abs(value - b) <= NUM_TOL for b in bank)


def score_record(rec: dict, tally: ArmTally) -> None:
    decision = rec.get("decision") or {}
    text = (decision.get("rationale") or rec.get("rationale") or "").strip()
    if not text:
        return
    tally.n_records += 1
    is_bid = decision.get("action") == "bid"
    if is_bid:
        tally.n_bids += 1
    profit_bank, edge_bank, ctx = _trace_numeric_bank(rec.get("tool_calls") or [])
    if ctx["has_llm_sim"]:
        tally.autonomous_sim_records += 1
    both_bank = profit_bank | edge_bank

    # 1. numeric grounding
    for m in RE_EUR_MWH.finditer(text):
        tally.nums_total += 1
        if _grounded(float(m.group(1)), edge_bank | profit_bank):
            tally.nums_grounded += 1
    for m in RE_EUR.finditer(text):
        # skip if this match was inside an EUR/MWh already counted (regex /MWh negative-lookahead handles it)
        tally.nums_total += 1
        if _grounded(float(m.group(1)), both_bank):
            tally.nums_grounded += 1

    # 2. acceptance claim — only scored on actual bids, negation-aware. A bid whose rationale
    # affirms acceptance is faithful iff an accepted sim matches the CHOSEN bid (side+qty+price).
    if is_bid and _affirms_acceptance(text):
        tally.accept_claims += 1
        if _bid_matches_accepted(decision, ctx["accepted_sims"]):
            tally.accept_supported += 1
        else:
            tally.accept_contradicted += 1
            if len(tally.examples_contradiction) < 5:
                tally.examples_contradiction.append(f"[{rec.get('agent_id')} s{rec.get('step')}] bid claims acceptance, no matching accepted sim: {text[:140]}")

    # 3. regime claim
    claimed_regimes = [w for w in REGIME_WORDS if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)]
    if claimed_regimes:
        tally.regime_claims += 1
        label = ctx["regime_label"]
        if label is None:
            tally.regime_unfounded += 1
            if len(tally.examples_unfounded) < 5:
                tally.examples_unfounded.append(f"[{rec.get('agent_id')} s{rec.get('step')}] regime claim, no regime tool: {text[:120]}")
        else:
            norm = label.replace("_watch", "").replace("_", " ")
            if any(cw.replace("_", " ") in (norm + " " + label) for cw in claimed_regimes):
                tally.regime_supported += 1
            else:
                tally.regime_contradicted += 1
                if len(tally.examples_contradiction) < 5:
                    tally.examples_contradiction.append(f"[{rec.get('agent_id')} s{rec.get('step')}] claims {claimed_regimes} but regime_label={label}: {text[:120]}")

    # 4. evidence-existence claims about outages / grid the agent never gathered
    if OUTAGE_RE.search(text):
        tally.evidence_claims += 1
        if not ctx["outage_called"]:
            tally.evidence_unfounded += 1
            if len(tally.examples_unfounded) < 8:
                tally.examples_unfounded.append(f"[{rec.get('agent_id')} s{rec.get('step')}] outage claim w/o get_outages: {text[:120]}")
    if GRID_RE.search(text):
        tally.evidence_claims += 1
        if not ctx["grid_called"]:
            tally.evidence_unfounded += 1


def score_run(run_dir: Path) -> ArmTally:
    tally = ArmTally()
    tpath = run_dir / "traces.jsonl"
    if not tpath.exists():
        return tally
    for line in tpath.read_text().splitlines():
        if line.strip():
            score_record(json.loads(line), tally)
    return tally


def arm_name(run_dir: Path) -> str:
    # lvar-<arm>-aprNN-... or lva-<arm>-...
    parts = run_dir.name.split("-")
    return parts[1] if len(parts) > 1 else run_dir.name


def _rate(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run dirs containing traces.jsonl")
    ap.add_argument("--glob", help="glob pattern for run dirs")
    ap.add_argument("--json-out", type=Path, help="write per-arm tallies to JSON")
    args = ap.parse_args()

    run_dirs: list[Path] = [Path(r) for r in args.runs]
    if args.glob:
        run_dirs += [Path(p) for p in glob(args.glob)]
    run_dirs = [d for d in run_dirs if (d / "traces.jsonl").exists()]
    if not run_dirs:
        raise SystemExit("no run dirs with traces.jsonl")

    arms: dict[str, ArmTally] = {}
    for d in sorted(run_dirs):
        arm = arm_name(d)
        t = score_run(d)
        agg = arms.setdefault(arm, ArmTally())
        for f in ("n_records", "n_bids", "nums_total", "nums_grounded", "accept_claims",
                  "accept_supported", "accept_contradicted", "regime_claims", "regime_supported",
                  "regime_contradicted", "regime_unfounded", "evidence_claims", "evidence_unfounded",
                  "autonomous_sim_records"):
            setattr(agg, f, getattr(agg, f) + getattr(t, f))
        agg.examples_contradiction = (agg.examples_contradiction + t.examples_contradiction)[:8]
        agg.examples_unfounded = (agg.examples_unfounded + t.examples_unfounded)[:8]

    print(f"\nRationale faithfulness vs trace — {len(run_dirs)} runs, {len(arms)} arms\n")
    hdr = f"{'arm':<16}{'recs':>5}{'bids':>5}{'num-grounded':>14}{'accept-halluc':>15}{'regime-acc':>12}{'regime-unfnd':>13}{'evid-unfnd':>11}{'autonomous':>11}"
    print(hdr); print("-" * len(hdr))
    out = {}
    for arm in sorted(arms):
        t = arms[arm]
        regime_den = t.regime_supported + t.regime_contradicted
        row = (
            f"{arm:<16}{t.n_records:>5}{t.n_bids:>5}"
            f"{_rate(t.nums_grounded, t.nums_total):>14}"
            f"{_rate(t.accept_contradicted, t.accept_claims):>15}"
            f"{_rate(t.regime_supported, regime_den):>12}"
            f"{_rate(t.regime_unfounded, t.regime_claims):>13}"
            f"{_rate(t.evidence_unfounded, t.evidence_claims):>11}"
            f"{_rate(t.autonomous_sim_records, t.n_records):>11}"
        )
        print(row)
        out[arm] = t.__dict__
    print("\nLegend: num-grounded=quoted EUR amounts exactly matching a trace tool value "
          "(unmatched = derived differences/sums OR fabricated — not separated here); "
          "accept-halluc=BIDS affirming acceptance with no accepted sim matching the chosen bid; "
          "regime-acc=regime claims matching the regime tool label; "
          "regime-unfnd/evid-unfnd=claims about evidence the agent never retrieved (no tool call); "
          "autonomous=records with >=1 LLM-requested simulate call (vs runner-seeded transcription).\n")
    for arm in sorted(arms):
        t = arms[arm]
        if t.examples_contradiction or t.examples_unfounded:
            print(f"### {arm} flagged examples")
            for e in t.examples_contradiction[:4]:
                print("  CONTRADICTED:", e)
            for e in t.examples_unfounded[:4]:
                print("  UNFOUNDED:   ", e)
            print()

    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
