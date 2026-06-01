from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.pypsa_adapter import (  # noqa: E402
    build_tiny_dk_network,
    export_heimdall_scenario_bundle,
    export_network,
    solve_network,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the tiny DK1/DK2 PyPSA scenario")
    parser.add_argument("--out", type=Path, default=Path("data/processed/pypsa/tiny-dk"))
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--write-netcdf", action="store_true")
    args = parser.parse_args()

    network = build_tiny_dk_network()
    solve = solve_network(network, solver_name=args.solver)
    if solve.status != "ok" or solve.condition != "optimal":
        raise SystemExit(f"PyPSA solve failed: {solve.status}/{solve.condition}")

    bundle = export_heimdall_scenario_bundle(network, args.out, source="tiny_pypsa_dk")
    if args.write_netcdf:
        export_network(network, args.out / "network.nc")
    print(f"scenario={bundle.scenario_path}")
    print(f"dispatch={bundle.dispatch_path}")
    print(f"manifest={bundle.manifest_path}")


if __name__ == "__main__":
    main()
