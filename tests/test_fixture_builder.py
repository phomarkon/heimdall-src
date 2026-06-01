from pathlib import Path

import pandas as pd

from packages.data import build_simulator_fixture, file_sha256


def test_build_simulator_fixture_is_real_dk_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "imbalance.parquet"
    rows = []
    for tick in pd.date_range("2025-03-04T00:00:00Z", periods=16, freq="15min"):
        for zone in ["DK1", "DK2"]:
            rows.append(
                {
                    "utc_timestamp": tick,
                    "zone": zone,
                    "satisfied_demand_mw": 10.0,
                    "imbalance_price_eur_mwh": 100.0,
                    "spot_price_eur_mwh": 90.0,
                    "mfrr_marginal_price_up_eur_mwh": 120.0,
                    "mfrr_marginal_price_down_eur_mwh": 80.0,
                }
            )
    pd.DataFrame(rows).to_parquet(source, index=False)

    first = build_simulator_fixture(source, tmp_path / "fixture_a.json", ticks=16)
    second = build_simulator_fixture(source, tmp_path / "fixture_b.json", ticks=16)

    assert first.tick_count == 16
    assert first.zones == ["DK1", "DK2"]
    assert first.is_real_data
    assert file_sha256(first.path) == file_sha256(second.path)
