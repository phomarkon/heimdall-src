from heimdall_contracts import PersonaArchetype
from heimdall_personas import default_persona


def test_default_persona_round_trip() -> None:
    p = default_persona("agent-007", PersonaArchetype.P2H)
    assert p.archetype is PersonaArchetype.P2H
    assert p.storage_mwh == 100.0
    assert p.forecaster_id == "F8"
