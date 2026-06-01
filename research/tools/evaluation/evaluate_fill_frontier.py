"""Counterfactual fill/profit frontier from the accepted-candidate menu (no new runs).

Answers: the deterministic baseline picks the best-MARGIN accepted candidate (max worst_case_profit),
priced high -> low fill. If instead you picked for CLEARING (most aggressive price) or for max EXPECTED
profit (the optimal margin×clear balance), how much higher is fill, at what profit cost? And where does
the LLM's actual choice land? Computed per agent-tick over the candidates already simulated in the
traces, scored against realized activation truth — so it isolates the SELECTION policy on identical menus.

Strategies (one accepted candidate per agent-tick):
  best_margin  = argmax worst_case_profit_eur          (what deterministic_best_accepted picks)
  max_expected = argmax expected_profit_eur            (the margin×clear optimum a smart det would pick)
  max_fill     = most aggressive price (up:lowest, down:highest)  (pure clearing-seeker)
  llm_actual   = the candidate the LLM actually submitted

Honest reading: if max_expected >> best_margin on fill at similar profit, the low-fill problem is fixed
by a better DETERMINISTIC policy — not uniquely by the LLM. The LLM adds fill value only if llm_actual
beats max_expected.

Usage:
    PYTHONPATH=. uv run python tools/evaluation/evaluate_fill_frontier.py \
        --glob 'ai-society/runs/d6-fill-20260524/d6-llmfill-*24-q32' --json-out evaluations/fill_frontier.json
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import pandas as pd

from tools.evaluation.evaluate_society_run import _load_truth


def _cands(rec: dict) -> list[dict]:
    out = []
    for tc in rec.get("tool_calls") or []:
        a = tc.get("arguments") or {}
        res = tc.get("result") or {}
        if not isinstance(res, dict) or res.get("accepted") is not True:
            continue
        if a.get("side") not in ("up", "down") or a.get("limit_price_eur_mwh") is None:
            continue
        out.append({"side": a["side"], "qty": float(a.get("quantity_mwh") or 0),
                    "price": float(a["limit_price_eur_mwh"]),
                    "wc": res.get("worst_case_profit_eur"), "exp": res.get("expected_profit_eur")})
    # dedupe identical (side,qty,price)
    seen = {}
    for c in out:
        seen[(c["side"], round(c["qty"], 3), round(c["price"], 3))] = c
    return list(seen.values())


def _truth_row(truth: pd.DataFrame, ts, zone):
    r = truth[(truth["timestamp_utc"] == pd.Timestamp(ts)) & (truth["zone"] == zone)]
    r = r[r["activation_direction"].isin(["up", "down"])]
    if r.empty:
        return None
    row = r.sort_values("activated_volume_mwh", ascending=False).iloc[0]
    return str(row["activation_direction"]), float(row["settlement_price_eur_mwh"]), float(row["spot_price_eur_mwh"]), float(row["activated_volume_mwh"])


def _fill_profit(c: dict, truth_row) -> tuple[int, float]:
    if truth_row is None:
        return 0, 0.0
    direction, settle, spot, vol = truth_row
    if c["side"] != direction or vol <= 0:
        return 0, 0.0
    crosses = (c["price"] <= settle) if c["side"] == "up" else (c["price"] >= settle)
    if not crosses:
        return 0, 0.0
    cleared = min(c["qty"], vol)
    ppm = (settle - spot) if c["side"] == "up" else (spot - settle)
    return 1, cleared * ppm


def _pick(cands, key):
    vals = [c for c in cands if c.get(key) is not None]
    return max(vals, key=lambda c: c[key]) if vals else None


def _pick_fill(cands):
    # most aggressive price = lowest for up, highest for down; mixed sides -> pick the side with more cands
    if not cands:
        return None
    ups = [c for c in cands if c["side"] == "up"]
    downs = [c for c in cands if c["side"] == "down"]
    side = ups if len(ups) >= len(downs) else downs
    return min(side, key=lambda c: c["price"]) if side is ups else max(side, key=lambda c: c["price"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True)
    ap.add_argument("--truth-dir", type=Path, default=Path("data/cache/evaluation_truth/april_2026"))
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    truth = _load_truth(args.truth_dir / "activation_truth.parquet")
    dirs = []
    for g in args.glob:
        dirs += [Path(p) for p in glob(g)]
    dirs = sorted({d for d in dirs if (d / "traces.jsonl").exists()})

    strat = {s: {"sel": 0, "fill": 0, "profit": 0.0} for s in ("best_margin", "max_expected", "max_fill", "llm_actual")}
    for d in dirs:
        for line in (d / "traces.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cands = _cands(r)
            if not cands:
                continue
            tr = _truth_row(truth, r["timestamp"], r.get("zone", "DK1"))
            picks = {"best_margin": _pick(cands, "wc"), "max_expected": _pick(cands, "exp"), "max_fill": _pick_fill(cands)}
            dec = r.get("decision") or {}
            if dec.get("action") == "bid" and dec.get("limit_price_eur_mwh") is not None:
                picks["llm_actual"] = {"side": dec.get("side"), "qty": float(dec.get("quantity_mwh") or 0), "price": float(dec["limit_price_eur_mwh"])}
            for s, c in picks.items():
                if c is None:
                    continue
                f, p = _fill_profit(c, tr)
                strat[s]["sel"] += 1
                strat[s]["fill"] += f
                strat[s]["profit"] += p

    out = {}
    print(f"\nFill/profit frontier (counterfactual on identical menus) — {len(dirs)} runs\n")
    print(f"{'strategy':<14}{'selected':>10}{'fill_rate':>11}{'total_profit':>14}{'profit/selected':>16}")
    for s in ("best_margin", "max_expected", "max_fill", "llm_actual"):
        a = strat[s]
        fr = a["fill"] / a["sel"] if a["sel"] else float("nan")
        pps = a["profit"] / a["sel"] if a["sel"] else float("nan")
        print(f"{s:<14}{a['sel']:>10}{fr:>11.3f}{a['profit']:>14.0f}{pps:>16.2f}")
        out[s] = {"selected": a["sel"], "fill_rate": round(fr, 4), "total_profit": round(a["profit"], 1), "profit_per_selected": round(pps, 3)}
    print("\nReading: compare best_margin (det baseline) vs max_expected (smart det) vs max_fill vs llm_actual.\n")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
