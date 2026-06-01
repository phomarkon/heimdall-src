from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.pypsa_adapter import load_heimdall_scenario  # noqa: E402
from packages.simulator import (  # noqa: E402
    Bid,
    ConstantBidPolicy,
    ReplaySimulator,
    write_simulation_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic Heimdall replay")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zone", default="DK1", choices=["DK1", "DK2"])
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--side", default="down", choices=["up", "down"])
    parser.add_argument("--submitted-at")
    parser.add_argument("--quantity-mwh", type=float, default=4.0)
    parser.add_argument("--limit-price", type=float, default=40.0)
    args = parser.parse_args()

    scenario = load_heimdall_scenario(args.scenario)
    bid = Bid(
        agent_id="cli-sample",
        asset_id=args.zone,
        zone=args.zone,
        utc_timestamp=args.timestamp,
        side=args.side,
        quantity_mwh=args.quantity_mwh,
        limit_price_eur_mwh=args.limit_price,
        submitted_at_utc=args.submitted_at,
    )
    result = ReplaySimulator.from_files(args.fixture, scenario).run(ConstantBidPolicy([bid]))
    write_simulation_trace(result, args.out, args.fixture)
    print(f"result={args.out}")
    print(f"hash={result.result_hash}")


if __name__ == "__main__":
    main()
