import json
from pathlib import Path

from packages.data import file_sha256
from packages.pypsa_adapter import (
    build_tiny_dk_network,
    export_heimdall_scenario_bundle,
    solve_network,
)


def test_scenario_bundle_extracts_physical_envelope_and_dispatch(tmp_path: Path) -> None:
    network = build_tiny_dk_network()
    solve_network(network, solver_name="highs")

    bundle = export_heimdall_scenario_bundle(
        network,
        tmp_path / "scenario",
        source="tiny_pypsa_dk",
    )

    scenario = json.loads((bundle.path / "scenario.json").read_text())
    dispatch = json.loads((bundle.path / "dispatch.json").read_text())
    manifest = json.loads((bundle.path / "manifest.json").read_text())

    assert scenario["zones"] == ["DK1", "DK2"]
    assert scenario["p2h_assets"]["DK1"]["p_nom_mw"] == 50.0
    assert scenario["p2h_assets"]["DK1"]["ramp_limit_mw_per_tick"] == 25.0
    assert scenario["physical_envelope"]["DK1"]["asset_capacity_mw"] == 50.0
    assert scenario["physical_envelope"]["DK1"]["ramp_mw_per_tick"] == 25.0
    assert scenario["physical_envelope"]["DK1"]["cop_profile"]["type"] == "constant"
    assert scenario["physical_envelope"]["DK1"]["thermal_loss_per_tick"] == 0.0001
    assert scenario["thermal_storage"]["DK1"]["e_nom_mwh"] == 100.0
    assert scenario["thermal_storage"]["DK1"]["initial_soc_mwh"] == 40.0
    assert scenario["ev_fleets"]["DK1"]["energy_mwh"] == 80.0
    assert scenario["wind_assets"]["DK1"]["p_nom_mw"] == 80.0
    assert scenario["generator_assets"]["DK1"]["marginal_cost_eur_mwh"] == 85.0
    assert scenario["renewables_assets"]["DK1"]["component_ids"] == ["DK1 wind"]
    assert scenario["retailer_assets"]["DK1"]["flexible_load_mw"] == 12.0
    assert scenario["asset_provenance"]["schema_version"] == "1.0.0"
    assert scenario["provenance"]["scenario_id"] == "dk1-dk2-p2h-v0"
    assert scenario["provenance"]["full_pypsa_eur_required"] is False
    assert scenario["interconnectors"]["DK1-DK2"]["capacity_mw"] == 600.0
    assert len(dispatch["rows"]) == 32
    assert {row["zone"] for row in dispatch["rows"]} == {"DK1", "DK2"}
    assert manifest["source"] == "tiny_pypsa_dk"
    assert manifest["scenario_id"] == "dk1-dk2-p2h-v0"
    assert manifest["provenance"]["full_pypsa_eur_required"] is False
    assert manifest["files"]["scenario.json"]["sha256"] == file_sha256(
        bundle.path / "scenario.json"
    )
    assert manifest["files"]["dispatch.json"]["schema_hash"]
