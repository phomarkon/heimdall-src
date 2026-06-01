import os

import pytest

from packages.data import (
    build_simulator_fixture,
    fetch_eds_dataset,
    normalize_eds_imbalance_price,
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_HEIMDALL_INTEGRATION") != "1",
        reason="Set RUN_HEIMDALL_INTEGRATION=1 to call public EDS endpoints",
    ),
]


def test_public_eds_imbalance_fixture_smoke(tmp_path) -> None:
    raw = fetch_eds_dataset(
        "ImbalancePrice",
        start="2025-03-04",
        end="2025-03-05",
        price_areas=["DK1", "DK2"],
    )
    normalized = normalize_eds_imbalance_price(raw)
    source = tmp_path / "imbalance.parquet"
    normalized.to_parquet(source, index=False)

    result = build_simulator_fixture(source, tmp_path / "fixture.json", ticks=16)

    assert result.tick_count == 16
    assert result.zones == ["DK1", "DK2"]
    assert normalized[normalized["zone"] == "DK1"]["utc_timestamp"].nunique() >= 16
