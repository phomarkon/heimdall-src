from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.data import fetch_eds_dataset, normalize_eds_imbalance_price  # noqa: E402
from packages.simulator import BaselineMFRRForecaster, backtest_price_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate calibrated Heimdall mFRR price-response engine"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Normalized EDS parquet input")
    source.add_argument("--fetch-eds", action="store_true", help="Fetch live EDS data")
    parser.add_argument("--start", default="2025-03-04")
    parser.add_argument("--end", default="2025-03-18")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--forecast-timestamp")
    parser.add_argument("--forecast-zone", choices=["DK1", "DK2"], default="DK1")
    parser.add_argument("--issued-at")
    args = parser.parse_args()

    if args.fetch_eds:
        raw = fetch_eds_dataset(
            "ImbalancePrice",
            start=args.start,
            end=args.end,
            price_areas=["DK1", "DK2"],
        )
        frame = normalize_eds_imbalance_price(raw)
    else:
        frame = pd.read_parquet(args.input)

    report = backtest_price_model(
        frame,
        train_fraction=args.train_fraction,
        min_samples=args.min_samples,
    )
    payload = {"backtest": asdict(report)}
    if args.forecast_timestamp and args.issued_at:
        forecaster = BaselineMFRRForecaster.fit(frame)
        payload["forecast"] = asdict(
            forecaster.forecast(
                delivery_timestamp=args.forecast_timestamp,
                zone=args.forecast_zone,
                issued_at=args.issued_at,
            )
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
