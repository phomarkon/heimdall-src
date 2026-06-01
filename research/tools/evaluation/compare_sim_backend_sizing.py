from __future__ import annotations

import json
from pathlib import Path
from typing import Any


WINDOWS = ("apr02-0530", "apr09-1830", "apr13-0015")
BACKENDS = {
    "scenario": {
        "medium_prefix": "tsa-s06-scenario-full",
        "new_prefix": "sbs-s06-scenario",
    },
    "pypsa": {
        "medium_prefix": "tsa-s06-pypsa-full",
        "new_prefix": "sbs-s06-pypsa",
    },
}
SIZINGS = ("current", "medium", "large")
OUTPUT_DIR = Path("evaluations/sim-backend-sizing-20260519")

METRICS = [
    "realized_profit_eur",
    "truth_window_oracle_capture",
    "bid_action_count",
    "watch_count",
    "wrong_side_count",
    "filled_count",
    "cleared_mwh",
    "asset_backend_disagreement_rate",
    "asset_backend_proxy_false_positive_rate",
    "asset_backend_scenario_envelope_false_positive_rate",
    "unsupported_bid_proposal_rate",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for backend_slug, backend in BACKENDS.items():
        for window in WINDOWS:
            medium = _summary(f"{backend['medium_prefix']}-{window}-seed42-q32")
            for sizing in SIZINGS:
                run_id = (
                    f"{backend['medium_prefix']}-{window}-seed42-q32"
                    if sizing == "medium"
                    else f"{backend['new_prefix']}-{sizing}-{window}-seed42-q32"
                )
                summary = _summary(run_id)
                rows.append(
                    {
                        "backend": backend_slug,
                        "window": window,
                        "sizing": sizing,
                        "run_id": run_id,
                        "metrics": {key: summary.get(key) for key in METRICS},
                        "delta_vs_medium": {key: _delta(summary.get(key), medium.get(key)) for key in METRICS},
                        "quantity_bucket_metrics": summary.get("quantity_bucket_metrics", {}),
                        "accepted_candidate_rate_by_quantity_bucket": summary.get(
                            "accepted_candidate_rate_by_quantity_bucket", {}
                        ),
                    }
                )
    payload = {"matrix": "sim-backend-sizing-20260519", "rows": rows}
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _summary(run_id: str) -> dict[str, Any]:
    path = Path("evaluations") / run_id / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _delta(left: Any, right: Any) -> float | None:
    try:
        return round(float(left) - float(right), 6)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
