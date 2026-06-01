import subprocess
import sys


def test_pypsa_and_simulator_clis_show_help() -> None:
    for command in [
        [sys.executable, "research/tools/pypsa/bootstrap_pypsa_eur.py", "--help"],
        [sys.executable, "research/tools/pypsa/export_tiny_scenario.py", "--help"],
        [sys.executable, "research/tools/simulator/run_replay.py", "--help"],
        [sys.executable, "research/tools/simulator/validate_mfrr_engine.py", "--help"],
    ]:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        assert "usage:" in result.stdout.lower()
