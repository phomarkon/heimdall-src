from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FixtureBuildResult:
    path: Path
    tick_count: int
    zones: list[str]
    is_real_data: bool


def _iso_z(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def build_simulator_fixture(
    source_path: Path, output_path: Path, *, ticks: int = 16
) -> FixtureBuildResult:
    frame = pd.read_parquet(source_path)
    required = {
        "utc_timestamp",
        "zone",
        "satisfied_demand_mw",
        "imbalance_price_eur_mwh",
        "spot_price_eur_mwh",
        "mfrr_marginal_price_up_eur_mwh",
        "mfrr_marginal_price_down_eur_mwh",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Fixture source missing required columns: {missing}")

    normalized = frame.copy()
    normalized["utc_timestamp"] = pd.to_datetime(
        normalized["utc_timestamp"], utc=True, errors="coerce"
    )
    normalized = normalized.sort_values(["utc_timestamp", "zone"]).reset_index(drop=True)
    tick_values = list(normalized["utc_timestamp"].drop_duplicates().head(ticks))
    if len(tick_values) != ticks:
        raise ValueError(f"Expected {ticks} unique ticks, found {len(tick_values)}")

    selected = normalized[normalized["utc_timestamp"].isin(tick_values)].copy()
    zones = sorted(selected["zone"].dropna().unique().tolist())
    payload = {
        "schema_version": "1.0.0",
        "source": "real_eds_imbalance_price",
        "tick_count": ticks,
        "zones": zones,
        "ticks": [
            {
                "utc_timestamp": _iso_z(pd.Timestamp(tick)),
                "markets": selected[selected["utc_timestamp"] == tick]
                .drop(columns=["utc_timestamp"])
                .sort_values("zone")
                .to_dict(orient="records"),
            }
            for tick in tick_values
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return FixtureBuildResult(
        path=output_path.resolve(),
        tick_count=ticks,
        zones=zones,
        is_real_data=payload["source"].startswith("real_"),
    )
