"""Outcome-debrief experiment (explainability axis #3 — "explain what happened").

The rationales in a run are ex-ante. This tests the ex-post axis: given a bid and the RAW
settlement facts (settlement price, spot, activation direction/volume — but NOT the derived
fill/profit), can the agent explain what happened — did it fill, did it profit, why? We score
the debrief against the realized ground truth (`bid_evaluations.parquet` from evaluate_society_run):
  - fill_correct   : states filled/not consistent with realized status
  - profitsign_correct : states profit/loss/zero consistent with realized_profit
  - confabulated   : asserts a fill/profit that contradicts the truth

Two arms on the SAME bids: the LLM debrief (Qwen3-32B) vs a deterministic template that derives
fill/profit from the settlement formula. If the template matches/beats the LLM (expected, per the
adversarial-template finding for ex-ante rationales), the debrief artifact does not require an LLM.

Usage:
  python tools/evaluation/run_outcome_debrief.py --eval-dir evaluations/<run> --run-dir <run> \
      --n 40 --base-url http://127.0.0.1:8000/v1 --out evaluations/outcome_debrief_<run>.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import requests

FILL_NEG = re.compile(r"(not fill|didn'?t fill|did not fill|wasn'?t fill|was not fill|unfilled|"
                      r"no fill|not clear|did not clear|didn'?t clear|not execut|not eligible|"
                      r"could not|couldn'?t|did not match|didn'?t match|no activation|"
                      r"not cross|did not cross|wrong side|opposite direction|cannot clear|can'?t clear)", re.I)
FILL_POS = re.compile(r"(did fill|was filled|the bid fill|bid filled|filled because|cleared because|"
                      r"was cleared|successfully (?:fill|clear)|did clear|was executed|got filled)", re.I)
PROFIT_POS = re.compile(r"\b(profit|gain|positive|earned|made money|favou?rable)\b", re.I)
PROFIT_NEG = re.compile(r"\b(loss|lost|negative|unprofitable|cost|out of the money)\b", re.I)
PROFIT_ZERO = re.compile(r"\b(zero|no profit|no loss|break[- ]?even|nothing|0 ?eur|€0)\b", re.I)


_THINK = re.compile(r"<think>.*?</think>", re.S)


def llm_debrief(base_url: str, key: str, model: str, prompt: str) -> str:
    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "temperature": 0.2, "max_tokens": 320,
              # Qwen3 thinking mode off: otherwise <think> eats the budget and truncates the answer.
              "chat_template_kwargs": {"enable_thinking": False},
              "messages": [
                  {"role": "system", "content": "You are a Nordic mFRR balancing-market trader writing a "
                   "short post-settlement debrief. State plainly whether the bid filled and whether it "
                   "made a profit or a loss, and why. 2-3 sentences. Do not invent numbers. /no_think"},
                  {"role": "user", "content": prompt}]},
        timeout=120,
    )
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"].strip()
    return _THINK.sub("", txt).strip()


def template_debrief(row: dict) -> str:
    status = row["status"]
    sp, spot = row.get("settlement_price_eur_mwh"), row.get("spot_price_eur_mwh")
    profit = row.get("realized_profit_eur") or 0.0
    filled = status in ("filled", "partially_filled")
    fillword = "filled" if filled else "did not fill"
    if profit > 1e-6:
        pword = f"a profit of {profit:.1f} EUR"
    elif profit < -1e-6:
        pword = f"a loss of {abs(profit):.1f} EUR"
    else:
        pword = "zero realized P&L"
    why = {
        "filled": "the limit price crossed the settlement price and the side matched the activation.",
        "partially_filled": "only part of the volume cleared at the settlement price.",
        "price_not_crossed": "the activation settled but the limit price did not cross it.",
        "wrong_side": "the market activated the opposite side, so the bid could not clear.",
        "no_activation": "there was no activation on this side, so nothing cleared.",
    }.get(status, "the bid did not clear under the realized settlement.")
    extra = f" Settlement {sp:.1f} vs spot {spot:.1f} EUR/MWh." if isinstance(sp, (int, float)) and isinstance(spot, (int, float)) else ""
    return f"The {row.get('side')} bid {fillword}, with {pword}; {why}{extra}"


def score(text: str, row: dict) -> dict:
    text = re.sub(r"[*_`#]", "", text)  # strip markdown so word patterns match
    status = row["status"]
    profit = row.get("realized_profit_eur") or 0.0
    truth_filled = status in ("filled", "partially_filled")
    # Read the verdict from the OPENING clause — every debrief starts "The bid (did not) fill...".
    head = text[:90].lower()
    says_nofill = bool(re.search(r"(not fill|did ?n.?t fill|not clear|did ?n.?t clear|not execut)", head))
    says_fill = (not says_nofill) and bool(re.search(r"(bid fill|bid was fill|filled|did fill|cleared|did clear)", head))
    if not (says_fill or says_nofill):  # fall back to whole text
        says_nofill = bool(FILL_NEG.search(text))
        says_fill = (not says_nofill) and bool(FILL_POS.search(text))
    fill_correct = (says_fill and truth_filled) or (says_nofill and not truth_filled)
    fill_stated = says_fill or says_nofill
    # profit sign (only meaningful when the bid actually filled)
    if profit > 1e-6:
        ps_correct = bool(PROFIT_POS.search(text)) and not bool(PROFIT_NEG.search(text))
    elif profit < -1e-6:
        ps_correct = bool(PROFIT_NEG.search(text))
    else:
        ps_correct = True  # zero P&L: not scored on profit wording
    # confabulation = asserts a FILL that did not happen (the clean definition)
    confab = says_fill and not truth_filled
    return {"fill_stated": fill_stated, "fill_correct": fill_correct,
            "profitsign_correct": ps_correct, "confabulated": confab}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, required=True, help="dir with bid_evaluations.parquet")
    ap.add_argument("--truth-dir", type=Path, required=True, help="dir with activation_truth.parquet")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key", default="heimdall-local")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bids = pd.read_parquet(args.eval_dir / "bid_evaluations.parquet")
    truth = pd.read_parquet(args.truth_dir / "activation_truth.parquet")
    truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True)
    bids["timestamp_utc"] = pd.to_datetime(bids["timestamp_utc"], utc=True)
    tcols = ["timestamp_utc", "zone", "activation_direction", "settlement_price_eur_mwh", "spot_price_eur_mwh"]
    bids = bids.merge(truth[tcols], on=["timestamp_utc", "zone"], how="left")
    bids = bids[bids["status"].isin(["filled", "partially_filled", "price_not_crossed", "wrong_side", "no_activation"])]
    # Stratify across the THREE outcome regimes so the metric is representative, not biased toward
    # the easy (direction-mismatch) cases: a fill, a same-side-but-price-not-crossed, and a wrong-side.
    strata = {
        "filled": bids[bids["status"].isin(["filled", "partially_filled"])],
        "price_not_crossed": bids[bids["status"] == "price_not_crossed"],
        "wrong_side": bids[bids["status"].isin(["wrong_side", "no_activation"])],
    }
    per = max(1, args.n // 3)
    sample = pd.concat([df.head(per) for df in strata.values()])
    arms = {"llm": [], "template": []}
    for _, r in sample.iterrows():
        row = r.to_dict()
        sp = row.get("settlement_price_eur_mwh"); spot = row.get("spot_price_eur_mwh")
        adir = row.get("activation_direction")
        # RAW settlement facts only — fill/profit are WITHHELD (the LLM must infer them).
        facts = (f"You submitted a {row.get('side')} bid of {row.get('quantity_mwh')} MWh at limit "
                 f"{row.get('limit_price_eur_mwh')} EUR/MWh. After delivery the realized market state was: "
                 f"the market activated the '{adir}' direction; settlement price {sp} EUR/MWh; "
                 f"spot price {spot} EUR/MWh. Clearing rule: a bid clears only if its side matches the "
                 f"activated direction AND its limit price <= the settlement price; profit per MWh when "
                 f"filled is (settlement - spot) for an up bid and (spot - settlement) for a down bid. "
                 f"Explain what happened to your bid and why — did it fill, and did it make a profit or a loss?")
        try:
            txt = llm_debrief(args.base_url, args.key, args.model, facts)
        except Exception as e:  # noqa: BLE001
            txt = f"[error {e}]"
        arms["llm"].append(score(txt, row))
        arms["template"].append(score(template_debrief(row), row))

    out = {}
    for arm, recs in arms.items():
        n = len(recs)
        out[arm] = {
            "n": n,
            "fill_correct": round(sum(x["fill_correct"] for x in recs) / n, 3) if n else 0,
            "profitsign_correct": round(sum(x["profitsign_correct"] for x in recs) / n, 3) if n else 0,
            "confabulated": sum(x["confabulated"] for x in recs),
            "fill_stated": round(sum(x["fill_stated"] for x in recs) / n, 3) if n else 0,
        }
    print(json.dumps(out, indent=2))
    args.out.write_text(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
