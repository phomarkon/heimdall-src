"""Build the leakage-safe RAG corpus for the Heimdall LLM society (2026-05-25).

Every document is sourced from real repository data — no invented facts — and is
stamped with a ``market_as_of`` instant on the *simulated* market clock. The
retriever only ever serves a document whose ``market_as_of`` is on/before the
decision tick (timeless methodology cards carry ``None``). That cutoff is what
keeps the experiment honest: a bid on a window can never retrieve that window's
realised outcome.

Three document classes:
  1. historical_stats   — per (day, zone) market-regime cards computed from
                          activation_truth + context_windows parquets
                          (March + April 2026). as_of = end of that day.
                          Plus a per-month overview card (as_of = month end).
  2. prior_run_lesson   — one card per evaluations/*/run_summary.json, tagged with
                          its window's simulated end as as_of, carrying the run's
                          realised capture / fill / profit.
  3. methodology        — timeless cards from ADR index + verifier/market/asset
                          /persona facts in the codebase. as_of = None.

Usage:
  PYTHONPATH=.:ai-society/src uv run python tools/experiments/build_rag_corpus_20260525.py
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from heimdall_ai_society.rag import RagDocument

REPO = Path(__file__).resolve().parents[3]
OUT_PATH = REPO / "ai-society" / "rag" / "corpus.jsonl"

CONTEXT_DIRS = {
    "2026-03": REPO / "data/cache/real_context/2026_03/context_windows.parquet",
    "2026-04": REPO / "data/cache/real_context/april_2026/context_windows.parquet",
}
TRUTH_DIRS = {
    "2026-03": REPO / "data/cache/evaluation_truth/2026_03/activation_truth.parquet",
    "2026-04": REPO / "data/cache/evaluation_truth/april_2026/activation_truth.parquet",
}
EVAL_GLOB = REPO / "evaluations"
ADR_INDEX = REPO / "docs/adr/INDEX.md"

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
WINDOW_RE = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d{2})-(\d{2})(\d{2})")
TICKS_RE = re.compile(r"-(\d+)-q\d+")


def _f(x: float | None, nd: int = 1) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.{nd}f}"


# --------------------------------------------------------------------------
# 1. historical market-regime cards
# --------------------------------------------------------------------------
def build_historical_cards() -> list[RagDocument]:
    docs: list[RagDocument] = []
    for month, truth_path in TRUTH_DIRS.items():
        if not truth_path.exists():
            print(f"[corpus] missing {truth_path}, skipping")
            continue
        truth = pd.read_parquet(truth_path)
        ctx = pd.read_parquet(CONTEXT_DIRS[month]) if CONTEXT_DIRS[month].exists() else None
        truth["timestamp_utc"] = pd.to_datetime(truth["timestamp_utc"], utc=True)
        truth["date"] = truth["timestamp_utc"].dt.date
        if ctx is not None:
            ctx["timestamp_utc"] = pd.to_datetime(ctx["timestamp_utc"], utc=True)
            ctx["date"] = ctx["timestamp_utc"].dt.date

        for (date, zone), g in truth.groupby(["date", "zone"]):
            g = g.sort_values("timestamp_utc")
            n = len(g)
            dir_counts = g["activation_direction"].value_counts().to_dict()
            up = int(dir_counts.get("up", 0))
            down = int(dir_counts.get("down", 0))
            neutral = int(dir_counts.get("neutral", 0))
            activated = g[g["activation_direction"].isin(["up", "down"])]
            settle = g["settlement_price_eur_mwh"]
            imb = g["imbalance_price_eur_mwh"]
            dominant = "up" if up > down else ("down" if down > up else "balanced")
            # 6-hour block tendency
            blocks = []
            for lo, hi, label in [(0, 6, "night"), (6, 12, "morning"), (12, 18, "afternoon"), (18, 24, "evening")]:
                blk = g[(g["timestamp_utc"].dt.hour >= lo) & (g["timestamp_utc"].dt.hour < hi)]
                if blk.empty:
                    continue
                bc = blk["activation_direction"].value_counts().to_dict()
                bup, bdown = int(bc.get("up", 0)), int(bc.get("down", 0))
                btend = "up" if bup > bdown else ("down" if bdown > bup else "flat")
                blocks.append(f"{label}={btend}(settle~{_f(blk['settlement_price_eur_mwh'].mean(),0)})")

            ctx_bits = ""
            if ctx is not None:
                cg = ctx[(ctx["date"] == date) & (ctx["zone"] == zone)]
                if not cg.empty:
                    ctx_bits = (
                        f" Day-ahead price mean {_f(cg['price_day_ahead_eur_mwh'].mean(), 0)} EUR/MWh; "
                        f"load mean {_f(cg['load_actual_mw'].mean(), 0)} MW; "
                        f"wind gen mean {_f(cg['generation_wind_mw'].mean(), 0)} MW."
                    )

            text = (
                f"Market regime {zone} on {date} (simulated mFRR, 15-min). "
                f"{n} MTUs: {up} up-activated, {down} down-activated, {neutral} neutral "
                f"(dominant side: {dominant}). "
                f"Settlement price mean {_f(settle.mean(), 0)}, p95 {_f(settle.quantile(0.95), 0)}, max {_f(settle.max(), 0)} EUR/MWh. "
                f"Imbalance price mean {_f(imb.mean(), 0)}, p95 {_f(imb.quantile(0.95), 0)} EUR/MWh. "
                f"Activated volume mean {_f(activated['activated_volume_mwh'].mean(), 1)} MWh over {len(activated)} activated MTUs. "
                f"Intraday tendency: {', '.join(blocks)}.{ctx_bits}"
            )
            as_of = datetime(date.year, date.month, date.day, 23, 59, 59, tzinfo=UTC)
            docs.append(RagDocument(
                doc_id=f"hist-{zone}-{date}",
                text=text,
                source=f"activation_truth+context_windows {month}",
                kind="historical_stats",
                market_as_of=as_of,
                metadata={"zone": zone, "date": str(date), "dominant_side": dominant,
                          "up_ticks": up, "down_ticks": down},
            ))

        # per-month, per-zone overview (as_of = month end)
        for zone, g in truth.groupby("zone"):
            up = int((g["activation_direction"] == "up").sum())
            down = int((g["activation_direction"] == "down").sum())
            settle = g["settlement_price_eur_mwh"]
            last_ts = g["timestamp_utc"].max()
            month_end = datetime(last_ts.year, last_ts.month, last_ts.day, 23, 59, 59, tzinfo=UTC)
            text = (
                f"Monthly overview {zone} {month}: {len(g)} MTUs, {up} up vs {down} down activations "
                f"(up share {up / max(1, up + down):.0%} of activated). "
                f"Settlement price mean {_f(settle.mean(), 0)}, p95 {_f(settle.quantile(0.95), 0)}, "
                f"max {_f(settle.max(), 0)} EUR/MWh. Higher settlement prices reward correctly-sided mFRR bids; "
                f"up-heavy months favour up-side (P2H load-shed) participation."
            )
            docs.append(RagDocument(
                doc_id=f"hist-overview-{zone}-{month}",
                text=text,
                source=f"activation_truth {month}",
                kind="historical_stats",
                market_as_of=month_end,
                metadata={"zone": zone, "month": month},
            ))
    return docs


# --------------------------------------------------------------------------
# 2. prior-run lesson cards
# --------------------------------------------------------------------------
def _window_from_run_id(run_id: str) -> tuple[datetime, str] | None:
    m = WINDOW_RE.search(run_id.lower())
    if not m:
        return None
    mon, dd, hh, mm = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    try:
        start = datetime(2026, MONTHS[mon], dd, hh, mm, tzinfo=UTC)
    except ValueError:
        return None
    return start, f"{mon}{dd:02d}-{hh:02d}{mm:02d}"


def build_run_lesson_cards() -> list[RagDocument]:
    docs: list[RagDocument] = []
    summaries = sorted(EVAL_GLOB.glob("*/run_summary.json"))
    skipped = 0
    for path in summaries:
        try:
            s = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue
        run_id = str(s.get("id") or path.parent.name)
        win = _window_from_run_id(run_id)
        if win is None:
            skipped += 1
            continue
        start, win_label = win
        tm = TICKS_RE.search(run_id)
        ticks = int(s.get("ticks") or (tm.group(1) if tm else 24))
        as_of = start + timedelta(minutes=15 * ticks)

        cap = s.get("opportunity_capture")
        pnl = s.get("cumulative_pnl_eur")
        oracle = s.get("oracle_feasible_profit_eur")
        fill = s.get("fill_rate")
        ppm = s.get("profit_per_mwh")
        bids = s.get("bid_count")
        filled = s.get("filled_count")
        if cap is None and pnl is None:
            skipped += 1
            continue
        text = (
            f"Past society run '{run_id}' on window {win_label} ({ticks} MTUs). "
            f"Opportunity capture {_f(cap, 3) if cap is not None else 'n/a'} "
            f"(realised PnL {_f(pnl, 0)} EUR vs submitted-bid oracle {_f(oracle, 0)} EUR). "
            f"Fill rate {_f(fill, 2)}, {filled}/{bids} bids filled, profit per MWh {_f(ppm, 1)} EUR. "
            f"Lesson: capture stays low because participation/sizing — not bid selection — is the binding lever; "
            f"correctly-sided, adequately-sized bids on activated MTUs drive realised profit."
        )
        docs.append(RagDocument(
            doc_id=f"run-{run_id}",
            text=text,
            source=str(path.relative_to(REPO)),
            kind="prior_run_lesson",
            market_as_of=as_of,
            metadata={"window": win_label, "ticks": ticks,
                      "opportunity_capture": cap, "fill_rate": fill},
        ))
    print(f"[corpus] run-lesson cards: {len(docs)} built, {skipped} skipped (no window/metrics)")
    return docs


# --------------------------------------------------------------------------
# 3. methodology cards (timeless; sourced from codebase facts)
# --------------------------------------------------------------------------
def build_methodology_cards() -> list[RagDocument]:
    cards: list[tuple[str, str, str]] = []  # (doc_id, source, text)

    # ADR index rows
    if ADR_INDEX.exists():
        for line in ADR_INDEX.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\|\s*(\d{4})\s*\|\s*(.+?)\s*\|\s*([\d-]+)\s*\|\s*(\w+)\s*\|", line)
            if m:
                num, title, date, status = m.groups()
                cards.append((
                    f"adr-{num}",
                    "docs/adr/INDEX.md",
                    f"Architecture decision ADR-{num} ({status}, {date}): {title}.",
                ))

    # Faithful methodology facts (from the chapters / contracts)
    cards.extend([
        ("method-market",
         "chapter 01",
         "Heimdall simulates the post-2025-03-04 Nordic 15-minute mFRR energy-activation balancing market "
         "for Danish zones DK1/DK2. Each MTU (market time unit) is 15 minutes. A bid has a side (up = deliver "
         "energy / reduce load; down = absorb energy / increase load), a quantity in MWh, and a limit price in "
         "EUR/MWh. A bid is activated only if its side matches the realised activation direction and its limit "
         "price crosses the settlement price; profit is the settlement spread times cleared volume."),
        ("method-asset",
         "apps/pypsa-scenario / chapter 04",
         "The focal agent operates a power-to-heat (P2H) asset rated 50 MW with multi-horizon thermal storage "
         "(~6.25 MWh usable per ADR scope), plus longer-horizon thermal buffering. Sizing must respect available "
         "headroom and state of charge; oversizing relative to feasible headroom produces rejected or unfilled bids."),
        ("method-verifier",
         "apps/verifier / chapter 03",
         "Every focal bid passes a two-stage verifier before submission: (1) physical feasibility against the "
         "asset model, and (2) a conformal worst-case-profit guard that rejects bids whose worst-case profit "
         "falls below a tunable floor tau. Accepted bids inherit conformal coverage on realised profit "
         "regardless of LLM errors (Theorem 1a split-CP finite-sample; Theorem 1b online ACI long-run)."),
        ("method-sizing",
         "ai-society sizing knobs",
         "Candidate bid sizing is controlled by sizing mode (current/medium/large). Larger sizing on correctly-"
         "sided activated MTUs raises realised profit roughly proportionally, but only up to feasible asset "
         "headroom; wrong-side or infeasible sizing is wasted. Participation rate (how often the agent bids on "
         "genuinely activated MTUs) is the dominant lever on opportunity capture."),
        ("method-regime-taxonomy",
         "regime taxonomy",
         "Market-regime shorthand: 'up-heavy' = most activated MTUs are up (scarcity, high settlement prices, "
         "favour up-side delivery); 'down-heavy' = surplus, down-side absorption pays; 'volatile' = high "
         "settlement-price p95/mean ratio with frequent side flips; 'quiet' = mostly neutral MTUs where "
         "abstaining avoids fee/penalty exposure. Read historical regime cards to anticipate the likely side."),
        ("method-archetypes",
         "ai-society/src/.../personas.py",
         "Society archetypes and indicative capacities: P2H (50 MW, storage, risk-averse, forecaster F8 — the "
         "focal); WIND (80 MW, neutral, F9); GENERATOR (220 MW, neutral, F7); RENEWABLES (100-160 MW, F9/F10); "
         "RETAILER (130-180 MW demand response, F11/F12); EV (20 MW virtual battery, storage, F1). Each archetype "
         "bids only through its own simulator path."),
        ("method-capture-metric",
         "tools/evaluation/evaluate_society_run.py",
         "Opportunity capture = realised profit / submitted-bid oracle profit, where the oracle is what a "
         "clairvoyant bidder would have earned on the same submitted candidates with perfect activation "
         "knowledge. Capture rises by bidding the correct side at adequate size on MTUs that actually activate, "
         "and by not wasting bids on neutral MTUs."),
    ])

    return [
        RagDocument(doc_id=i, text=t, source=src, kind="methodology", market_as_of=None)
        for (i, src, t) in cards
    ]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    docs = build_methodology_cards() + build_historical_cards() + build_run_lesson_cards()
    # de-dup by doc_id (keep first)
    seen: set[str] = set()
    unique: list[RagDocument] = []
    for d in docs:
        if d.doc_id in seen:
            continue
        seen.add(d.doc_id)
        unique.append(d)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for d in unique:
            fh.write(json.dumps(d.to_json(), sort_keys=True) + "\n")
    by_kind: dict[str, int] = {}
    for d in unique:
        by_kind[d.kind] = by_kind.get(d.kind, 0) + 1
    print(f"[corpus] wrote {len(unique)} documents to {OUT_PATH}")
    print(f"[corpus] by kind: {by_kind}")


if __name__ == "__main__":
    main()
