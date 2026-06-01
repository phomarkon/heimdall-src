import json
from pathlib import Path

from packages.simulator import Bid, ConstantBidPolicy, ReplaySimulator


def _fixture(path: Path) -> Path:
    payload = {
        "schema_version": "1.0.0",
        "source": "real_eds_imbalance_price",
        "tick_count": 16,
        "zones": ["DK1", "DK2"],
        "ticks": [
            {
                "utc_timestamp": f"2025-03-04T{hour:02d}:{minute:02d}:00Z",
                "markets": [
                    {
                        "zone": "DK1",
                        "imbalance_price_eur_mwh": 80 + index,
                        "spot_price_eur_mwh": 50,
                        "mfrr_marginal_price_up_eur_mwh": 120,
                        "mfrr_marginal_price_down_eur_mwh": 20,
                        "satisfied_demand_mw": 100,
                    },
                    {
                        "zone": "DK2",
                        "imbalance_price_eur_mwh": 75 + index,
                        "spot_price_eur_mwh": 49,
                        "mfrr_marginal_price_up_eur_mwh": 115,
                        "mfrr_marginal_price_down_eur_mwh": 19,
                        "satisfied_demand_mw": 90,
                    },
                ],
            }
            for index, (hour, minute) in enumerate(
                [(0, 0), (0, 15), (0, 30), (0, 45), (1, 0), (1, 15), (1, 30), (1, 45),
                 (2, 0), (2, 15), (2, 30), (2, 45), (3, 0), (3, 15), (3, 30), (3, 45)]
            )
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_replay_combines_real_fixture_with_pypsa_envelope(tmp_path: Path, tiny_dk_scenario) -> None:
    simulator = ReplaySimulator.from_files(_fixture(tmp_path / "fixture.json"), tiny_dk_scenario)
    policy = ConstantBidPolicy(
        [
            Bid(
                agent_id="focal",
                asset_id="DK1",
                zone="DK1",
                utc_timestamp="2025-03-04T00:00:00Z",
                side="down",
                quantity_mwh=4.0,
                limit_price_eur_mwh=40.0,
            )
        ]
    )

    result = simulator.run(policy)

    assert result.tick_count == 16
    assert result.zones == ["DK1", "DK2"]
    assert len(result.accepted_bids) == 1
    assert result.accepted_bids[0].settlement_eur == 120.0
    assert result.final_asset_states["DK1"].thermal_soc_mwh == 51.996
    assert result.result_hash


def test_replay_rejects_bad_timestamp_and_is_deterministic(tmp_path: Path, tiny_dk_scenario) -> None:
    fixture = _fixture(tmp_path / "fixture.json")
    scenario = tiny_dk_scenario
    policy = ConstantBidPolicy(
        [
            Bid(
                agent_id="focal",
                asset_id="DK1",
                zone="DK1",
                utc_timestamp="2025-03-04T00:15:00Z",
                side="down",
                quantity_mwh=4.0,
                limit_price_eur_mwh=40.0,
            )
        ]
    )

    first = ReplaySimulator.from_files(fixture, scenario).run(policy)
    second = ReplaySimulator.from_files(fixture, scenario).run(policy)

    assert len(first.rejected_bids) == 1
    assert first.rejected_bids[0].reason_code == "wrong_tick"
    assert first.result_hash == second.result_hash
