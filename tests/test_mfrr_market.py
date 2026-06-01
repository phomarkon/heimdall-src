import json
from pathlib import Path

from packages.simulator import Bid, ConstantBidPolicy, MFRRMarketClock, ReplaySimulator
from packages.simulator.trace import write_simulation_trace


def _fixture(path: Path) -> Path:
    payload = {
        "schema_version": "1.0.0",
        "source": "real_eds_imbalance_price",
        "tick_count": 16,
        "zones": ["DK1", "DK2"],
        "ticks": [
            {
                "utc_timestamp": f"2025-03-04T{idx//4:02d}:{(idx%4)*15:02d}:00Z",
                "markets": [
                    {
                        "zone": "DK1",
                        "imbalance_price_eur_mwh": 80,
                        "spot_price_eur_mwh": 50,
                        "mfrr_marginal_price_up_eur_mwh": 120,
                        "mfrr_marginal_price_down_eur_mwh": 20,
                        "satisfied_demand_mw": 100,
                    },
                    {
                        "zone": "DK2",
                        "imbalance_price_eur_mwh": 75,
                        "spot_price_eur_mwh": 49,
                        "mfrr_marginal_price_up_eur_mwh": 115,
                        "mfrr_marginal_price_down_eur_mwh": 19,
                        "satisfied_demand_mw": 90,
                    },
                ],
            }
            for idx in range(16)
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_mfrr_clock_uses_45_minute_submission_gate_and_7_5_minute_acceptance_notice(
    tmp_path: Path,
    tiny_dk_scenario,
) -> None:
    accepted_bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T00:00:00Z",
        side="down",
        quantity_mwh=4.0,
        limit_price_eur_mwh=40.0,
        submitted_at_utc="2025-03-03T23:14:00Z",
    )
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T00:00:00Z",
        side="down",
        quantity_mwh=4.0,
        limit_price_eur_mwh=40.0,
        submitted_at_utc="2025-03-03T23:16:00Z",
    )

    clock = MFRRMarketClock(submission_gate_minutes=45.0, acceptance_notice_minutes=7.5)
    accepted = ReplaySimulator.from_files(
        _fixture(tmp_path / "accepted.json"),
        tiny_dk_scenario,
        clock=clock,
    ).run(ConstantBidPolicy([accepted_bid]))
    rejected = ReplaySimulator.from_files(
        _fixture(tmp_path / "rejected.json"),
        tiny_dk_scenario,
        clock=clock,
    ).run(ConstantBidPolicy([bid]))

    assert len(accepted.accepted_bids) == 1
    assert accepted.accepted_bids[0].accepted_at_utc.isoformat().replace(
        "+00:00", "Z"
    ) == "2025-03-03T23:52:30Z"
    assert len(rejected.rejected_bids) == 1
    assert rejected.rejected_bids[0].reason_code == "gate_closed"


def test_mfrr_bid_book_clears_feasible_bid_and_trace_manifest_is_deterministic(
    tmp_path: Path,
    tiny_dk_scenario,
) -> None:
    fixture = _fixture(tmp_path / "fixture.json")
    scenario = tiny_dk_scenario
    bid = Bid(
        agent_id="focal",
        asset_id="DK1",
        zone="DK1",
        utc_timestamp="2025-03-04T00:00:00Z",
        side="down",
        quantity_mwh=4.0,
        limit_price_eur_mwh=40.0,
        submitted_at_utc="2025-03-03T23:14:00Z",
    )

    first = ReplaySimulator.from_files(fixture, scenario).run(ConstantBidPolicy([bid]))
    second = ReplaySimulator.from_files(fixture, scenario).run(ConstantBidPolicy([bid]))
    first_trace = write_simulation_trace(first, tmp_path / "trace-a.json", fixture)
    second_trace = write_simulation_trace(second, tmp_path / "trace-b.json", fixture)

    assert len(first.accepted_bids) == 1
    assert first.accepted_bids[0].clearing_market == "mFRR"
    assert first.accepted_bids[0].settlement_eur == 120.0
    assert first.result_hash == second.result_hash
    assert first_trace.manifest["result_hash"] == second_trace.manifest["result_hash"]
    assert first_trace.manifest["source_fixture_sha256"]
