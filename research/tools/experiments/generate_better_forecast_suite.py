from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path("ai-society/configs/better-forecast-suite")
RUNS_ROOT = Path("ai-society/runs/better-forecast-suite")
CONTEXT_DIR = Path("data/cache/real_context/april_2026")
DATA_CACHE_DIR = Path("data/cache/real_context/april_2026/source_cache")
FORECASTERS = ("f0", "f7", "f8", "f3_ensemble")


@dataclass(frozen=True)
class Window:
    slug: str
    start: str
    ticks: int
    reason: str


KNOWN_WINDOWS = (
    Window("apr02-1200", "2026-04-02T12:00:00Z", 24, "strong profit and good side precision"),
    Window("apr02-0530", "2026-04-02T05:30:00Z", 48, "strong profit and many actions"),
    Window("apr03-1430", "2026-04-03T14:30:00Z", 48, "new profitable screened run"),
    Window("apr09-1830", "2026-04-09T18:30:00Z", 48, "best absolute profit"),
    Window("apr13-0015", "2026-04-13T00:15:00Z", 48, "huge independent oracle and mostly watch-only"),
    Window("apr06-1300", "2026-04-06T13:00:00Z", 48, "high watch recall and no bid conversion"),
    Window("apr22-0830", "2026-04-22T08:30:00Z", 48, "smaller profitable sanity window"),
    Window("apr05-1030", "2026-04-05T10:30:00Z", 48, "small-profit sanity window"),
)

BID_WINDOWS = tuple(window for window in KNOWN_WINDOWS if window.slug in {"apr02-1200", "apr02-0530", "apr03-1430", "apr09-1830"})


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {
        "smoke": [],
        "watch_study": [],
        "bid_conversion": [],
        "no_llm": [],
        "rolling_eval": [],
    }

    for forecaster in ("f7", "f8", "f3_ensemble"):
        written["smoke"].append(write_config("smoke", KNOWN_WINDOWS[0], forecaster, 0.0, "llm"))
    written["smoke"].append(write_config("smoke", KNOWN_WINDOWS[0], "f8", 0.0, "deterministic_best_accepted"))

    for window in KNOWN_WINDOWS:
        for forecaster in FORECASTERS:
            written["watch_study"].append(write_config("watch-study", window, forecaster, 0.0, "llm"))

    for window in BID_WINDOWS:
        for forecaster in FORECASTERS:
            for tau in (0.0, -50.0, -100.0):
                written["bid_conversion"].append(write_config("bid-conversion", window, forecaster, tau, "llm"))

    for window in BID_WINDOWS:
        for forecaster in FORECASTERS:
            written["no_llm"].append(write_config("no-llm", window, forecaster, 0.0, "deterministic_best_accepted"))

    rolling_windows = select_rolling_windows()
    for window in rolling_windows:
        written["rolling_eval"].append(write_config("rolling-eval", window, "f8", 0.0, "llm"))
        written["rolling_eval"].append(write_config("rolling-eval", window, "f3_ensemble", 0.0, "llm"))
        written["rolling_eval"].append(write_config("rolling-eval", window, "f8", 0.0, "deterministic_best_accepted"))

    write_manifest(written, rolling_windows)
    write_batch_scripts(written)


def write_config(group: str, window: Window, forecaster: str, tau: float, chooser: str) -> str:
    chooser_slug = "llm-q32" if chooser == "llm" else "det"
    tau_slug = f"tau{int(tau)}".replace("-", "m")
    group_slug = {
        "smoke": "smoke",
        "watch-study": "watch",
        "bid-conversion": "bid",
        "no-llm": "detbase",
        "rolling-eval": "roll",
    }[group]
    run_id = f"da-{group_slug}-{window.slug}-{window.ticks}-{forecaster}-{tau_slug}-{chooser_slug}"
    path = ROOT / group / f"{run_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "seed": 42,
        "zone": "DK1",
        "agent_count": 5,
        "archetype_cycle": ["p2h", "wind", "generator", "renewables", "retailer"],
        "ticks": window.ticks,
        "start_timestamp": window.start,
        "forecaster_backend": forecaster,
        "chooser_mode": chooser,
        "verifier_mode": "simulator",
        "verifier_tau_eur": tau,
        "market_context": "real",
        "tool_mode": "openai_tools",
        "objective": "bid_seeking",
        "ablation_strategy": "diverse_action_society",
        "persona_profile": "diverse_action",
        "scenario_id": "p2h_dk1_pypsa",
        "tool_policy": "p2h_only_simulator",
        "max_tool_rounds": 6,
        "data_start": "2026-04-01T00:00:00Z",
        "data_end": "2026-05-01T00:00:00Z",
        "context_dataset_dir": str(CONTEXT_DIR),
        "data_cache_dir": str(DATA_CACHE_DIR),
        "default_lookback_hours": 24,
        "cache_refresh": False,
        "output_dir": "ai-society/runs",
        "llm": {
            "enabled": chooser == "llm",
            "model": "Qwen/Qwen3-32B",
            "temperature": 0.2,
            "max_tokens": 900,
            "timeout_seconds": 180,
            "max_concurrency": 1,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return str(path)


def select_rolling_windows() -> list[Window]:
    prices = pd.read_parquet(CONTEXT_DIR / "prices.parquet")
    prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
    dk1 = prices[prices["zone"] == "DK1"].copy()
    pivot = dk1.pivot_table(index="timestamp_utc", columns="price_type", values="price_eur_mwh", aggfunc="last").sort_index()
    spot = pivot.get("day_ahead", pd.Series(dtype=float)).ffill()
    imbalance = pivot.get("imbalance", spot).fillna(spot)
    up = pivot.get("mfrr_up", imbalance).fillna(imbalance)
    down = pivot.get("mfrr_down", imbalance).fillna(imbalance)
    candidates = []
    starts = pd.date_range("2026-04-02T00:00:00Z", "2026-04-29T00:00:00Z", freq="15min")
    for start in starts:
        hist_start = start - pd.Timedelta(hours=24)
        hist = pd.DataFrame({"spot": spot, "imbalance": imbalance, "up": up, "down": down})
        hist = hist[(hist.index >= hist_start) & (hist.index < start)]
        if len(hist) < 32:
            continue
        up_spread = (hist["up"] - hist["spot"]).dropna()
        down_spread = (hist["spot"] - hist["down"]).dropna()
        volatility = float(hist["imbalance"].diff().abs().tail(16).mean()) if len(hist) > 1 else 0.0
        up_positive = float((up_spread > 0).mean()) if not up_spread.empty else 0.0
        down_positive = float((down_spread > 0).mean()) if not down_spread.empty else 0.0
        recent_spread = float(up_spread.tail(8).mean()) if not up_spread.tail(8).empty else 0.0
        score = max(0.0, min(1.0, 0.45 * max(up_positive, down_positive) + 0.35 * min(volatility / 25.0, 1.0) + 0.20 * min(abs(recent_spread) / 50.0, 1.0)))
        candidates.append((score, start))
    selected: list[tuple[float, pd.Timestamp]] = []
    for score, start in sorted(candidates, reverse=True):
        if all(abs((start - prev).total_seconds()) >= 24 * 3600 for _, prev in selected):
            selected.append((score, start))
        if len(selected) == 6:
            break
    return [
        Window(f"roll{idx + 1}-{start.strftime('%b%d-%H%M').lower()}", start.strftime("%Y-%m-%dT%H:%M:%SZ"), 48, f"pre-registered agent-context score={score:.6f}")
        for idx, (score, start) in enumerate(selected)
    ]


def write_manifest(written: dict[str, list[str]], rolling_windows: list[Window]) -> None:
    manifest = {
        "schema_version": "1.0.0",
        "context_dataset_dir": str(CONTEXT_DIR),
        "known_windows": [window.__dict__ for window in KNOWN_WINDOWS],
        "rolling_selection_rule": "Rank non-overlapping 48-tick April DK1 windows using only prior 24h agent-context price spreads and volatility; no evaluation truth.",
        "rolling_windows": [window.__dict__ for window in rolling_windows],
        "config_counts": {key: len(value) for key, value in written.items()},
        "configs": written,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_batch_scripts(written: dict[str, list[str]]) -> None:
    def script(name: str, configs: list[str]) -> None:
        path = RUNS_ROOT / f"{name}.sh"
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cd /home/ucloud/heimdall\n"
            "uv run python ai-society/run_ablation_batch.py --expected-model Qwen/Qwen3-32B \\\n  "
            + " \\\n  ".join(configs)
            + "\n",
            encoding="utf-8",
        )
        path.chmod(0o755)

    script("run_smoke", written["smoke"])
    script("run_watch_study_a", written["watch_study"][::2])
    script("run_watch_study_b", written["watch_study"][1::2])
    script("run_bid_ablation_a", written["bid_conversion"][::2])
    script("run_bid_ablation_b", written["bid_conversion"][1::2])
    script("run_det_baseline", written["no_llm"])
    script("run_rolling_eval", written["rolling_eval"])


if __name__ == "__main__":
    main()
