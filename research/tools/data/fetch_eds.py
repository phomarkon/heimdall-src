from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.data import (
    EDS_DATASET_URL,
    fetch_eds_dataset,
    normalize_eds_day_ahead_prices,
    normalize_eds_imbalance_price,
    write_manifest,
)


NORMALIZERS = {
    "DayAheadPrices": normalize_eds_day_ahead_prices,
    "ImbalancePrice": normalize_eds_imbalance_price,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Energi Data Service datasets")
    parser.add_argument("dataset", choices=sorted(NORMALIZERS))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", default="data/raw/eds")
    args = parser.parse_args()

    raw = fetch_eds_dataset(
        args.dataset, start=args.start, end=args.end, price_areas=["DK1", "DK2"]
    )
    normalized = NORMALIZERS[args.dataset](raw)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset.lower()}_{args.start}_{args.end}.parquet"
    normalized.to_parquet(output_path, index=False)
    write_manifest(
        artifact_path=output_path,
        source_url=f"{EDS_DATASET_URL}/{args.dataset}",
        dataset=args.dataset,
        window_start_utc=args.start,
        window_end_utc=args.end,
        row_count=len(normalized),
        schema_columns=list(normalized.columns),
    )
    print(output_path)


if __name__ == "__main__":
    main()
