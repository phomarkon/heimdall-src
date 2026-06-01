"""Re-score completed AI-society runs with fairer denominators and a downside model.

Non-destructive: reads each run's ``traces.jsonl`` plus the shared evaluation truth,
reuses the canonical scoring from ``evaluate_society_run.py`` (so the clearing logic is
never forked), and recomputes:

  1a. ``oracle_capacity`` — THE HEADLINE FAIR ORACLE (P2H-FOCAL). Per-tick:
      max(0, spread) * min(real system mFRR activation volume, P2H focal capacity). Every input
      is grounded: the activation volume is real (Energinet ``TotalmFRRUp/DownMW`` x 0.25), the
      spread is real (mFRR settlement minus day-ahead spot), and the power figure is the P2H
      RAMP-limited envelope (ramp_limit_mw_per_tick x 0.25 = 6.25 MWh/MTU for p2h_dk1_pypsa) --
      the same physical ceiling the simulator enforces, not nominal p_nom. Capture is measured on the
      verifier-guarded P2H focal agent(s) alone (``capture_capacity = realized_p2h /
      oracle_capacity``), so it stays grounded even inside a heterogeneous society (the thesis
      framing: one focal P2H market-maker among heterogeneous BRP competitors). Non-P2H
      archetypes have assumed MW and are excluded from the denominator, never fabricated;
      grounded iff the run has >=1 P2H agent. For a P2H-only run this equals the society-wide
      capture. Independent of what the agents bid, so NOT gameable by timidity (unlike
      ``oracle_submitted``).
      CAVEAT (honest): the per-MTU cap ignores intertemporal storage/ramp coupling, so it is
      a LOOSE (conservative) upper bound — true achievable is <= oracle_capacity, hence true
      capture >= capture_capacity. Over short, sporadically-activated windows the SoC
      constraint rarely binds; a tight intertemporal-LP oracle is a possible follow-up.
  1b. ``oracle_submitted`` — per-tick oracle volume capped at the volume the agents actually
      offered on the activated side (best capturable *over the bids placed*). DIAGNOSTIC ONLY:
      it conflates "I chose not to bid" with "there was no opportunity", so it rewards
      timidity and must not be used as the headline value denominator. ``oracle_uncapped``
      (full system activation volume, physically unreachable by one society) is kept for
      reference as the loosest bound.
  1c. baseline-relative profit — paired deltas ``treatment - deterministic`` by
      (scenario, window, seed); an oracle-free comparison.

  SENSITIVITY ANALYSES (NOT headline value metrics — magnitudes are not data-grounded):
  - ``penalized_profit`` — realized profit minus ``lambda * wrong_side_mwh * spread``. lambda
    is a free dial, so this is a direction-error SENSITIVITY only; it is not a value metric
    (a wrong-side limit order that does not clear costs nothing in reality, and its true cost
    — missed right-side capture — is already reflected in ``capture_capacity``). lambda=0
    recovers the status quo.
  - ``delivery_adjusted_profit`` — realized profit minus a delivery shortfall: realized
    availability is a seeded, mean-preserving Beta draw around expected ``availability_share``
    a* (from the run's traces); the undelivered part of each filled bid is re-settled at the
    real ``imbalance_price_eur_mwh``. The re-settlement MECHANISM is real but the availability
    CV is an assumed magnitude, so this is a labelled execution-risk SENSITIVITY pending real
    availability data. Inert for firm assets (p2h/generator: no availability gate). CV=0
    recovers the status quo exactly.

Writes a JSON report; does not touch original run dirs or ``evaluations/``.

Usage:
    uv run python tools/evaluation/rescore_runs.py \
        --runs-dir ai-society/runs/chooser-det-llm-20260522 \
        --truth-dir data/cache/evaluation_truth/april_2026 \
        --out evaluations/rescore-chooser-det-llm-20260522/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from heimdall_contracts import PersonaArchetype
from heimdall_personas.archetypes import ARCHETYPE_DEFAULTS

# Reuse the canonical scorer rather than reimplementing the clearing rule.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_society_run as ev

# Capacity-capped "fair oracle": the maximum balancing profit reachable with perfect
# direction+price foresight, bounded by (i) the REAL system mFRR activation volume at the MTU
# (Energinet TotalmFRRUp/DownMW x 0.25) and (ii) the society's REAL physical power. Every input
# is grounded: only the per-agent power needs a spec, and P2H is the one archetype with a
# data-grounded one (PyPSA-Eur-Sec DK1 p_nom = 50 MW). So capture_capacity is reported as
# grounded ONLY for P2H-only societies; for invented-capacity archetypes (generator/renewables/
# retailer/wind/ev) it is left null rather than fabricated. This denominator is independent of
# what the agents bid, so it is not gameable by timidity (unlike oracle_submitted).
MTU_HOURS = 0.25
P2H_CAPACITY_MW = float(ARCHETYPE_DEFAULTS[PersonaArchetype.P2H]["capacity_mw"])  # 50.0 nominal p_nom (reference)
# The grounded per-MTU deliverable is the RAMP-limited scenario envelope, not nominal capacity:
# the simulator caps a P2H bid at ramp_limit_mw_per_tick * 0.25h (packages/simulator/
# pypsa_background.py). For p2h_dk1_pypsa: ramp_limit = 25 MW/tick -> 6.25 MWh/MTU (vs the 12.5
# a naive 50 MW x 0.25h would give). Sourced from scenario.p2h_assets[DK1].ramp_limit_mw_per_tick;
# pinned by test_p2h_ramp_constant_matches_scenario.
P2H_RAMP_LIMIT_MW_PER_TICK = 25.0
GROUNDED_CAPACITY_ARCHETYPES = {"p2h"}

CONDITIONS = ["deterministic", "guarded", "shadow-toolvisible"]
BASELINE_CONDITION = "deterministic"
# cdl-<scenario>-<condition>-<window>-seed<N>-...
RUN_ID_RE = re.compile(
    r"^cdl-(?P<scenario>.+?)-(?P<condition>deterministic|guarded|shadow-toolvisible)-"
    r"(?P<window>apr\d{2}-\d{4})-seed(?P<seed>\d+)-"
)
NON_BID_STATUSES = {"abstain", "watch", "invalid", "gate_closed", "missing_truth", "eligible"}
DEFAULT_LAMBDAS = [0.0, 0.25, 0.5, 1.0]
# Delivery-shortfall downside (Phase 1). CV = coefficient of variation of realized
# availability around its expected share a*; CV=0 recovers the status quo exactly.
DEFAULT_CVS = [0.0, 0.10, 0.15, 0.25]
FILLED_STATUSES = {"filled", "partially_filled"}


def _find_availability(obj: Any) -> float | None:
    """Recursively find an ``availability_share`` value (it sits in the simulate result's
    nested ``projected_state``/``next_state``, not at the top level)."""
    if isinstance(obj, dict):
        if "availability_share" in obj:
            try:
                return float(obj["availability_share"])
            except (TypeError, ValueError):
                pass
        for value in obj.values():
            found = _find_availability(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_availability(item)
            if found is not None:
                return found
    return None


def _availability_by_archetype(traces_path: Path) -> dict[str, float]:
    """Derive the decision-time expected availability share a* per archetype from the
    simulate tool-call results recorded in the run's own traces. Archetypes whose
    simulator does not gate on availability (generator, p2h) never record the field and
    are treated as firm (no delivery risk) by their absence here."""
    shares: dict[str, list[float]] = {}
    for line in traces_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or '"availability_share"' not in line:
            continue
        payload = json.loads(line)
        archetype = str(payload.get("archetype") or "")
        for call in payload.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            share = _find_availability(call.get("result"))
            if share is not None:
                shares.setdefault(archetype, []).append(share)
    return {arch: float(np.median(vals)) for arch, vals in shares.items() if vals}


def _realized_availability(*, run_seed: int, ts: Any, zone: str, side: str, agent_id: str,
                           archetype: str, cv: float, a_star: float) -> float:
    """Seeded, order-independent draw of realized availability rho with E[rho]=a_star.
    cv<=0 (or firm a_star>=1) returns a_star exactly, so the status quo is recovered."""
    if cv <= 0.0 or a_star >= 1.0 or a_star <= 0.0:
        return a_star
    kappa = (1.0 - a_star) / (a_star * cv * cv) - 1.0
    if kappa <= 0.0:  # cv too large for this mean; clamp to a wide-but-valid Beta
        kappa = 1e-3
    key = f"{run_seed}|{ts}|{zone}|{side}|{agent_id}|{archetype}|{cv}"
    sub_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(sub_seed)
    return float(rng.beta(a_star * kappa, (1.0 - a_star) * kappa))


def _shortfall_settlement(*, side: str, cleared: float, a_star: float, rho: float,
                          settlement: float, imbalance: float) -> tuple[float, float]:
    """Return (shortfall_mwh, penalty_eur) for a filled bid. You cannot deliver more than
    committed, so realized delivery is cleared * min(1, rho/a*); the undelivered part is
    re-settled at the imbalance price instead of the settlement price. penalty>0 is a loss,
    penalty<0 a windfall (when the imbalance price is favorable)."""
    if a_star <= 0.0 or a_star >= 1.0 or cleared <= 0.0:
        return 0.0, 0.0
    deliverable = cleared * min(1.0, rho / a_star)
    shortfall = max(0.0, cleared - deliverable)
    if shortfall <= 0.0:
        return 0.0, 0.0
    adverse_gap = (imbalance - settlement) if side == "up" else (settlement - imbalance)
    return shortfall, shortfall * adverse_gap


def _tick_truth(truth_rows: pd.DataFrame) -> tuple[str, float, float, float, float]:
    """Return (direction, activated_volume, realized_spread, settlement, imbalance) for a tick."""
    if truth_rows.empty:
        return "neutral", 0.0, 0.0, 0.0, 0.0
    activated = truth_rows[truth_rows["activation_direction"].isin(["up", "down"])]
    if activated.empty:
        return "neutral", 0.0, 0.0, 0.0, 0.0
    row = activated.sort_values("activated_volume_mwh", ascending=False).iloc[0]
    spread = ev._truth_profit_per_mwh(row)
    volume = max(0.0, float(row["activated_volume_mwh"]))
    settlement = float(row["settlement_price_eur_mwh"])
    imbalance = row.get("imbalance_price_eur_mwh")
    imbalance = settlement if imbalance is None or pd.isna(imbalance) else float(imbalance)
    return str(row["activation_direction"]), volume, float(spread), settlement, imbalance


def _p2h_focal_capacity_mwh_per_mtu(bids: pd.DataFrame) -> tuple[float, bool, int]:
    """P2H-FOCAL capacity-oracle input: the per-MTU physical energy the verifier-guarded P2H
    focal agent(s) can deliver. P2H is the single archetype with a data-grounded asset spec
    (PyPSA-Eur-Sec DK1 p_nom = 50 MW), so capture is measured on the P2H focal alone — this
    keeps the oracle grounded even inside a heterogeneous society (the thesis framing: one
    focal P2H market-maker among heterogeneous BRP competitors). Non-P2H archetypes have
    assumed MW and are excluded from the denominator, never fabricated.

    Per-MTU deliverable is the RAMP-limited envelope (ramp_limit_mw_per_tick * 0.25), the same
    physical ceiling the simulator enforces, NOT nominal p_nom * 0.25.

    Returns (p2h_capacity_mwh_per_mtu, grounded, n_p2h_agents). grounded is True iff the run
    contains at least one P2H agent.
    """
    if bids.empty or "archetype" not in bids.columns or "agent_id" not in bids.columns:
        return 0.0, False, 0
    agents = bids[["agent_id", "archetype"]].dropna().drop_duplicates()
    agents = agents.assign(archetype=agents["archetype"].astype(str).str.lower())
    n_p2h = int((agents["archetype"].isin(GROUNDED_CAPACITY_ARCHETYPES)).sum())
    return n_p2h * P2H_RAMP_LIMIT_MW_PER_TICK * MTU_HOURS, n_p2h >= 1, n_p2h


def _per_run_metrics(bids: pd.DataFrame, truth: pd.DataFrame, lambdas: list[float],
                     cvs: list[float], a_star_by_archetype: dict[str, float],
                     run_seed: int) -> dict[str, Any]:
    realized = float(bids["realized_profit_eur"].astype(float).sum()) if not bids.empty else 0.0
    p2h_cap, capacity_grounded, n_p2h_agents = _p2h_focal_capacity_mwh_per_mtu(bids)
    realized_p2h = (
        float(bids[bids["archetype"].astype(str).str.lower() == "p2h"]["realized_profit_eur"].astype(float).sum())
        if (not bids.empty and "archetype" in bids.columns)
        else 0.0
    )
    oracle_uncapped = 0.0
    oracle_submitted = 0.0
    oracle_capacity = 0.0
    wrong_side_penalty_unit = 0.0  # sum of wrong_side_mwh * spread, scaled by lambda later
    wrong_side_mwh_total = 0.0
    # Delivery-shortfall accumulators, per CV. penalty is net (loss>0 minus windfall);
    # loss is the downside tail only (what a risk-averse desk actually cares about).
    shortfall_penalty = {cv: 0.0 for cv in cvs}
    shortfall_loss = {cv: 0.0 for cv in cvs}
    shortfall_mwh = {cv: 0.0 for cv in cvs}

    for (ts, zone), tick in bids.groupby(["timestamp_utc", "zone"], sort=False):
        truth_rows = truth[(truth["timestamp_utc"] == ts) & (truth["zone"] == zone)]
        direction, volume, spread, settlement, imbalance = _tick_truth(truth_rows)
        if direction == "neutral" or spread <= 0 or volume <= 0:
            # Wrong-side into a neutral tick is scored "no_activation" upstream -> no penalty.
            continue
        # Volume offered on the correct (activated) side at this tick.
        correct = tick[(tick["side"] == direction) & (~tick["status"].isin(NON_BID_STATUSES))]
        submitted_correct = float(pd.to_numeric(correct["quantity_mwh"], errors="coerce").fillna(0.0).sum())
        oracle_uncapped += spread * volume
        oracle_submitted += spread * min(volume, submitted_correct)
        oracle_capacity += spread * min(volume, p2h_cap)
        # Wrong-side commitments: agent bid the opposite of the activated direction.
        wrong = tick[tick["status"] == "wrong_side"]
        wrong_mwh = float(pd.to_numeric(wrong["quantity_mwh"], errors="coerce").fillna(0.0).sum())
        wrong_side_mwh_total += wrong_mwh
        wrong_side_penalty_unit += wrong_mwh * spread
        # Delivery shortfall: each filled bid on the activated side may under-deliver.
        for _, bid in tick[tick["status"].isin(FILLED_STATUSES)].iterrows():
            cleared = float(pd.to_numeric(bid.get("cleared_mwh"), errors="coerce") or 0.0)
            a_star = a_star_by_archetype.get(str(bid.get("archetype")), 1.0)
            if cleared <= 0.0 or a_star >= 1.0 or a_star <= 0.0:
                continue
            for cv in cvs:
                rho = _realized_availability(run_seed=run_seed, ts=ts, zone=zone, side=str(bid["side"]),
                                             agent_id=str(bid["agent_id"]), archetype=str(bid["archetype"]),
                                             cv=cv, a_star=a_star)
                sf, penalty = _shortfall_settlement(side=str(bid["side"]), cleared=cleared, a_star=a_star,
                                                    rho=rho, settlement=settlement, imbalance=imbalance)
                shortfall_penalty[cv] += penalty
                shortfall_loss[cv] += max(0.0, penalty)
                shortfall_mwh[cv] += sf

    out: dict[str, Any] = {
        "realized_profit_eur": round(realized, 4),
        "oracle_uncapped_eur": round(oracle_uncapped, 4),
        "oracle_submitted_eur": round(oracle_submitted, 4),
        "oracle_capacity_eur": round(oracle_capacity, 4) if capacity_grounded else None,
        "capture_uncapped": round(realized / oracle_uncapped, 6) if oracle_uncapped > 0 else None,
        "capture_submitted": round(realized / oracle_submitted, 6) if oracle_submitted > 0 else None,
        # P2H-focal: numerator is the focal P2H agent(s)' realized profit, denominator their
        # grounded capacity oracle. Identical to society-wide for a P2H-only run.
        "capture_capacity": round(realized_p2h / oracle_capacity, 6) if (capacity_grounded and oracle_capacity > 0) else None,
        "capacity_grounded": bool(capacity_grounded),
        "realized_p2h_eur": round(realized_p2h, 4),
        "p2h_capacity_mwh_per_mtu": round(p2h_cap, 4),
        "n_p2h_agents": int(n_p2h_agents),
        "wrong_side_mwh": round(wrong_side_mwh_total, 4),
        "wrong_side_count": int((bids["status"] == "wrong_side").sum()) if not bids.empty else 0,
        "filled_count": int((bids["status"] == "filled").sum()) if not bids.empty else 0,
    }
    for lam in lambdas:
        out[f"penalized_profit_eur_lambda{lam}"] = round(realized - lam * wrong_side_penalty_unit, 4)
    for cv in cvs:
        out[f"delivery_adjusted_profit_eur_cv{cv}"] = round(realized - shortfall_penalty[cv], 4)
        out[f"shortfall_loss_eur_cv{cv}"] = round(shortfall_loss[cv], 4)
        out[f"shortfall_mwh_cv{cv}"] = round(shortfall_mwh[cv], 4)
    return out


def _extract_seed(run_id: str) -> int:
    m = re.search(r"seed(\d+)", run_id)
    return int(m.group(1)) if m else 42


def _score_run(run_dir: Path, truth: pd.DataFrame, lambdas: list[float],
               cvs: list[float]) -> dict[str, Any] | None:
    traces_path = run_dir / "traces.jsonl"
    if not traces_path.exists() or traces_path.stat().st_size == 0:
        return None
    traces = ev._load_traces(traces_path)
    if traces.empty:
        return None
    bids = ev._score_bids(traces, truth)
    a_star_by_archetype = _availability_by_archetype(traces_path)
    metrics = _per_run_metrics(bids, truth, lambdas, cvs, a_star_by_archetype, _extract_seed(run_dir.name))
    metrics["run_id"] = run_dir.name
    metrics["a_star_by_archetype"] = {k: round(v, 4) for k, v in sorted(a_star_by_archetype.items())}
    return metrics


def _parse_run_id(run_id: str) -> dict[str, str] | None:
    m = RUN_ID_RE.match(run_id)
    if not m:
        return None
    return m.groupdict()


def _paired_deltas(rows: list[dict[str, Any]], lambdas: list[float], cvs: list[float]) -> dict[str, Any]:
    """Pair each treatment condition against deterministic by (scenario, window, seed)."""
    indexed: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        parsed = _parse_run_id(row["run_id"])
        if parsed is None:
            continue
        key = (parsed["scenario"], parsed["window"], parsed["seed"])
        indexed.setdefault(key, {})[parsed["condition"]] = row

    metrics_to_pair = (
        ["realized_profit_eur"]
        + [f"penalized_profit_eur_lambda{lam}" for lam in lambdas]
        + [f"delivery_adjusted_profit_eur_cv{cv}" for cv in cvs]
    )
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        if condition == BASELINE_CONDITION:
            continue
        per_metric: dict[str, Any] = {}
        for metric in metrics_to_pair:
            wins = ties = losses = 0
            deltas: list[float] = []
            pairs: list[dict[str, Any]] = []
            for key, conds in sorted(indexed.items()):
                if BASELINE_CONDITION not in conds or condition not in conds:
                    continue
                base_v = conds[BASELINE_CONDITION].get(metric)
                treat_v = conds[condition].get(metric)
                if base_v is None or treat_v is None:
                    continue
                delta = round(treat_v - base_v, 4)
                deltas.append(delta)
                pairs.append({"scenario": key[0], "window": key[1], "delta": delta,
                              "treatment": treat_v, "baseline": base_v})
                if delta > 1e-6:
                    wins += 1
                elif delta < -1e-6:
                    losses += 1
                else:
                    ties += 1
            per_metric[metric] = {
                "n_pairs": len(deltas),
                "treatment_wins": wins,
                "ties": ties,
                "baseline_wins": losses,
                "mean_delta_eur": round(sum(deltas) / len(deltas), 4) if deltas else None,
                "sum_delta_eur": round(sum(deltas), 4) if deltas else None,
                "pairs": pairs,
            }
        summary[condition] = per_metric
    return summary


def _condition_totals(rows: list[dict[str, Any]], lambdas: list[float], cvs: list[float]) -> dict[str, Any]:
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        parsed = _parse_run_id(row["run_id"])
        if parsed is None:
            continue
        by_cond.setdefault(parsed["condition"], []).append(row)
    out: dict[str, Any] = {}
    for cond, items in by_cond.items():
        tot_real = sum(r["realized_profit_eur"] for r in items)
        tot_oracle_unc = sum(r["oracle_uncapped_eur"] for r in items)
        tot_oracle_sub = sum(r["oracle_submitted_eur"] for r in items)
        grounded = [r for r in items if r.get("capacity_grounded") and r.get("oracle_capacity_eur")]
        tot_oracle_cap = sum(r["oracle_capacity_eur"] for r in grounded)
        tot_real_grounded = sum(r.get("realized_p2h_eur", 0.0) for r in grounded)
        entry = {
            "n_runs": len(items),
            "total_realized_eur": round(tot_real, 2),
            "capture_uncapped_volwt": round(tot_real / tot_oracle_unc, 6) if tot_oracle_unc > 0 else None,
            "capture_submitted_volwt": round(tot_real / tot_oracle_sub, 6) if tot_oracle_sub > 0 else None,
            "n_runs_capacity_grounded": len(grounded),
            "capture_capacity_volwt": round(tot_real_grounded / tot_oracle_cap, 6) if tot_oracle_cap > 0 else None,
            "total_wrong_side_mwh": round(sum(r["wrong_side_mwh"] for r in items), 2),
            "total_wrong_side_count": sum(r["wrong_side_count"] for r in items),
        }
        for lam in lambdas:
            entry[f"total_penalized_eur_lambda{lam}"] = round(
                sum(r[f"penalized_profit_eur_lambda{lam}"] for r in items), 2
            )
        for cv in cvs:
            entry[f"total_delivery_adjusted_eur_cv{cv}"] = round(
                sum(r[f"delivery_adjusted_profit_eur_cv{cv}"] for r in items), 2
            )
            entry[f"total_shortfall_loss_eur_cv{cv}"] = round(
                sum(r[f"shortfall_loss_eur_cv{cv}"] for r in items), 2
            )
            entry[f"total_shortfall_mwh_cv{cv}"] = round(sum(r[f"shortfall_mwh_cv{cv}"] for r in items), 2)
        out[cond] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True, help="Matrix dir holding per-run subdirs.")
    parser.add_argument("--truth-dir", type=Path, default=Path("data/cache/evaluation_truth/april_2026"))
    parser.add_argument("--glob", default="cdl-*-24-q32", help="Subdir glob (excludes smoke by default).")
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--cvs", type=float, nargs="+", default=DEFAULT_CVS,
                        help="Realized-availability CV sweep for the delivery-shortfall downside.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    truth = ev._load_truth(args.truth_dir / "activation_truth.parquet")
    run_dirs = sorted(d for d in args.runs_dir.glob(args.glob) if d.is_dir())
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for run_dir in run_dirs:
        metrics = _score_run(run_dir, truth, args.lambdas, args.cvs)
        if metrics is None:
            skipped.append(run_dir.name)
            continue
        rows.append(metrics)

    report = {
        "runs_dir": str(args.runs_dir),
        "truth_dir": str(args.truth_dir),
        "lambdas": args.lambdas,
        "cvs": args.cvs,
        "n_runs_scored": len(rows),
        "n_runs_skipped": len(skipped),
        "skipped": skipped,
        "condition_totals": _condition_totals(rows, args.lambdas, args.cvs),
        "paired_deltas": _paired_deltas(rows, args.lambdas, args.cvs),
        "per_run": sorted(rows, key=lambda r: r["run_id"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"scored {len(rows)} runs, skipped {len(skipped)} -> {args.out}")


if __name__ == "__main__":
    main()
