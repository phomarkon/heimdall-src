"""D1 event-window selector — find society windows where a material, freshly-published
outage with decision-relevant free text is active, so the LLM has an information edge to
exploit that a price-trained forecaster cannot cleanly attribute.

This answers the gating question for D1: are there enough outage *events* (not persistent
background outages already priced in) in the data we have to run a fair event study?

A window is a candidate iff an outage:
  - touches DK1/DK2 directly or via an interconnector pair containing DK1/DK2,
  - has max_unavailable_capacity_mw >= --min-mw,
  - is ACTIVE during the 6h (24-tick) window, and
  - is FRESH: published within --fresh-hours before the window start (a newly-announced
    forced outage is the LLM-exploitable case; a months-old persistent one is priced in).

Window start = next 15-min boundary at max(published_at, time_start) — the first moment an
agent could both know about it and have it active.

Usage:
    python tools/experiments/select_outage_event_windows.py \
        --outages data/cache/real_context/april_2026/outages.json \
        --data-start 2026-04-01 --data-end 2026-05-01
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

DK = {"DK1", "DK2"}
FORCED_KW = ("failure", "forced", "unplanned", "fault", "trip")
PLANNED_KW = ("maintenance", "overhaul", "foreseen", "planned", "construction", "work")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def _ceil_15(dt: datetime) -> datetime:
    minute = (dt.minute // 15 + (1 if dt.minute % 15 or dt.second else 0)) * 15
    return dt.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)


def classify(title: str) -> str:
    t = title.lower()
    if any(k in t for k in FORCED_KW):
        return "forced"
    if any(k in t for k in PLANNED_KW):
        return "planned"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outages", type=Path, default=Path("data/cache/real_context/april_2026/outages.json"))
    ap.add_argument("--data-start", default="2026-04-01")
    ap.add_argument("--data-end", default="2026-05-01")
    ap.add_argument("--min-mw", type=float, default=300.0)
    ap.add_argument("--fresh-hours", type=float, default=36.0)
    ap.add_argument("--window-hours", type=float, default=6.0)
    args = ap.parse_args()

    data_start = _parse(args.data_start + "T00:00:00Z")
    data_end = _parse(args.data_end + "T00:00:00Z")
    win = timedelta(hours=args.window_hours)

    recs = json.loads(args.outages.read_text())
    candidates = []
    for r in recs:
        zones = set(r.get("zones") or [])
        dk_relevant = bool(zones & DK)  # direct or interconnector pair containing a DK zone
        if not dk_relevant:
            continue
        cap = float(r.get("max_unavailable_capacity_mw") or 0.0)
        if cap < args.min_mw:
            continue
        t_start = _parse(r["time_start_utc"])
        t_end = _parse(r["time_end_utc"])
        pub = _parse(r["published_at_utc"])
        w0 = _ceil_15(max(pub, t_start))
        w1 = w0 + win
        # active during window and window inside the data we have
        if not (t_start <= w1 and t_end >= w0):
            continue
        if not (data_start <= w0 and w1 <= data_end):
            continue
        fresh = (w0 - pub) <= timedelta(hours=args.fresh_hours)
        candidates.append({
            "window_start": w0.strftime("%Y-%m-%dT%H:%MZ"),
            "zones": sorted(zones),
            "capacity_mw": cap,
            "kind": classify(r["title"]),
            "fresh": fresh,
            "published_at": pub.strftime("%Y-%m-%dT%H:%MZ"),
            "title": r["title"].splitlines()[0][:80],
        })

    candidates.sort(key=lambda c: (not c["fresh"], -c["capacity_mw"]))
    fresh_n = sum(c["fresh"] for c in candidates)
    print(f"\nD1 outage-event windows in [{args.data_start}, {args.data_end}) "
          f"(DK-relevant, >={args.min_mw:.0f}MW, active in {args.window_hours:.0f}h window)\n")
    print(f"  total candidate windows: {len(candidates)}   of which FRESH (<= {args.fresh_hours:.0f}h since publish): {fresh_n}\n")
    print(f"{'window_start':<18}{'cap':>7}  {'kind':<8}{'fresh':<6}{'zones':<16}title")
    print("-" * 100)
    for c in candidates:
        print(f"{c['window_start']:<18}{c['capacity_mw']:>7.0f}  {c['kind']:<8}{'yes' if c['fresh'] else 'no':<6}"
              f"{','.join(c['zones']):<16}{c['title']}")
    print()
    if fresh_n < 3:
        print("VERDICT: data-starved for a fair D1 event study on this outage pull — only "
              f"{fresh_n} fresh DK-relevant event window(s). Need a broader outage pull across the "
              "test window, or accept these as smoke-only and report the negative honestly.\n")
    else:
        print(f"VERDICT: {fresh_n} fresh event windows — viable for an event study at this threshold.\n")


if __name__ == "__main__":
    main()
