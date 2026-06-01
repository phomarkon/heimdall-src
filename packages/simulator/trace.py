from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

from packages.data import file_sha256

from .models import SimulationResult
from .replay import result_to_dict, write_result


@dataclass(frozen=True)
class TraceArtifact:
    trace_path: Path
    manifest_path: Path
    manifest: dict


def write_simulation_trace(
    result: SimulationResult,
    trace_path: Path,
    source_fixture_path: Path,
) -> TraceArtifact:
    write_result(result, trace_path)
    manifest_path = trace_path.with_suffix(f"{trace_path.suffix}.manifest.json")
    payload = result_to_dict(result)
    manifest = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "result_hash": result.result_hash,
        "source_fixture_path": str(source_fixture_path.resolve()),
        "source_fixture_sha256": file_sha256(source_fixture_path),
        "trace_sha256": file_sha256(trace_path),
        "tick_count": result.tick_count,
        "zones": result.zones,
        "accepted_bid_count": len(payload["accepted_bids"]),
        "rejected_bid_count": len(payload["rejected_bids"]),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return TraceArtifact(
        trace_path=trace_path.resolve(),
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
    )
