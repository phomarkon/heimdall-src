"""Pull the full DK1 history into the tidy panel store.

Idempotent: skips months whose parquet already exists. Per
docs/RESEARCH-PROPOSAL.md §5.1 the canonical window is 2020-01-01 → 2026-04-30.

Run:
    uv run python tools/fetch_dk1_history.py [--start 2020-01-01] [--end 2026-04-30]

Tokens are read from the environment (ENTSOE_API_TOKEN); never logged.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from dateutil.relativedelta import relativedelta

from heimdall_data.loaders import PROCESSED_ROOT, load_dk1_panel


def _month_starts(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    out: list[tuple[datetime, datetime]] = []
    while cur < end:
        nxt = cur + relativedelta(months=1)
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--out", type=Path, default=PROCESSED_ROOT)
    args = parser.parse_args(argv)

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    args.out.mkdir(parents=True, exist_ok=True)

    months = _month_starts(start, end)
    n_skipped = 0
    n_pulled = 0
    for m_start, m_end in months:
        out_path = args.out / f"dk1_panel_{m_start:%Y%m%d}_{m_end:%Y%m%d}.parquet"
        if out_path.exists():
            n_skipped += 1
            continue
        try:
            load_dk1_panel(m_start, m_end, cache_dir=args.out)
            n_pulled += 1
        except Exception as exc:  # noqa: BLE001 -- best-effort idempotent loop
            # DEVIATION: live API can rate-limit. We log and continue rather than
            # halt, so the script can be re-run to fill gaps.
            print(f"[fetch_dk1] {m_start:%Y-%m} skipped on error: {exc}", file=sys.stderr)
    print(f"[fetch_dk1] months requested={len(months)} pulled={n_pulled} skipped={n_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
