import json
from pathlib import Path

import pandas as pd

from packages.data import write_manifest


def test_write_manifest_contains_required_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "imbalance.parquet"
    pd.DataFrame(
        [{"utc_timestamp": "2025-03-04T22:30:00Z", "zone": "DK1"}]
    ).to_parquet(artifact, index=False)

    manifest_path = write_manifest(
        artifact_path=artifact,
        source_url="https://api.energidataservice.dk/dataset/ImbalancePrice",
        dataset="ImbalancePrice",
        window_start_utc="2025-03-04T00:00:00Z",
        window_end_utc="2025-03-05T00:00:00Z",
        row_count=1,
        schema_columns=["utc_timestamp", "zone"],
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_url"] == "https://api.energidataservice.dk/dataset/ImbalancePrice"
    assert manifest["dataset"] == "ImbalancePrice"
    assert manifest["window_start_utc"] == "2025-03-04T00:00:00Z"
    assert manifest["window_end_utc"] == "2025-03-05T00:00:00Z"
    assert manifest["row_count"] == 1
    assert manifest["schema_hash"]
    assert manifest["file_sha256"]
    assert manifest["artifact_path"].endswith("imbalance.parquet")
