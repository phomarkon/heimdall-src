from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.data import build_simulator_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-derived simulator fixture")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ticks", type=int, default=16)
    args = parser.parse_args()

    result = build_simulator_fixture(
        Path(args.source), Path(args.output), ticks=args.ticks
    )
    print(result.path)


if __name__ == "__main__":
    main()
