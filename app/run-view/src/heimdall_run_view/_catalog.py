"""Run catalog, catalog-entry cache, and setup/window metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from heimdall_run_view._utils import (
    RunRecord,
    _file_sha256,
    _iso_z,
    _optional_float,
    _optional_int,
    _read_first_jsonl,
    _read_run_summary_for_root,
    _scan_trace_metadata,
    default_repo_root,
)

# ---------------------------------------------------------------------------
# Constants shared across adapter sub-modules
# ---------------------------------------------------------------------------

FALLBACK_ARCHETYPES = [
    "wind",
    "ev",
    "retailer",
    "p2h",
    "generator",
    "arbitrageur",
    "grid-info",
    "outage-info",
    "price-info",
    "sizing-info",
    "uncertainty-info",
    "decision-info",
    "risk-info",
]
FALLBACK_RISKS = ["averse", "neutral", "seeking"]
FALLBACK_SOPHISTICATION = ["low", "medium", "high"]
FOCAL_AGENT_ID = "agent-000"
GRAPH_ARCHETYPE_ORDER = [
    "p2h",
    "ev",
    "generator",
    "wind",
    "retailer",
    "arbitrageur",
    "grid-info",
    "outage-info",
    "price-info",
    "sizing-info",
    "uncertainty-info",
    "decision-info",
    "risk-info",
]

# ---------------------------------------------------------------------------
# Per-file catalog cache
# ---------------------------------------------------------------------------

_CatalogEntry = dict[str, Any]
_FileStamp = tuple[int, int]  # (mtime_ns, size)
_CATALOG_CACHE: dict[str, dict[str, tuple[_FileStamp, _CatalogEntry]]] = {}


class RunCatalog:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_repo_root()

    def list_runs(self) -> list[dict[str, Any]]:
        key = str(self.root)
        memo = _CATALOG_CACHE.get(key)
        if memo is None:
            memo = self._read_cache_file()
            _CATALOG_CACHE[key] = memo
        entries: list[_CatalogEntry] = []
        seen: set[str] = set()
        dirty = False
        for trace_path in self.root.glob("research/llm/ai-society/runs/**/traces.jsonl"):
            path_str = str(trace_path)
            try:
                stat = trace_path.stat()
            except OSError:
                continue
            seen.add(path_str)
            stamp: _FileStamp = (stat.st_mtime_ns, stat.st_size)
            cached = memo.get(path_str)
            if cached is not None and cached[0] == stamp:
                entries.append(cached[1])
                continue
            entry = self._catalog_entry(trace_path)
            if entry is None:
                continue
            memo[path_str] = (stamp, entry)
            entries.append(entry)
            dirty = True
        stale = set(memo) - seen
        if stale:
            for path_str in stale:
                memo.pop(path_str, None)
            dirty = True
        if dirty:
            self._write_cache_file(memo)
        entries.sort(key=lambda item: item["run_id"])
        return entries

    def _catalog_entry(self, trace_path: Path) -> _CatalogEntry | None:
        record = self._record_for_trace(trace_path)
        if record is None:
            return None
        return {
            "run_id": record.run_id,
            "total_steps": record.total_steps,
            "trace_sha256": record.trace_sha256,
            "status": record.status,
            "trace_path": str(record.trace_path),
            **self._run_catalog_metadata(record),
        }

    def _cache_file(self) -> Path:
        return self.root / "research" / "llm" / "ai-society" / ".run-catalog-cache.json"

    def _read_cache_file(self) -> dict[str, tuple[_FileStamp, _CatalogEntry]]:
        path = self._cache_file()
        if not path.exists():
            return {}
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        files = blob.get("files") if isinstance(blob, dict) else None
        if not isinstance(files, dict):
            return {}
        memo: dict[str, tuple[_FileStamp, _CatalogEntry]] = {}
        for path_str, record in files.items():
            try:
                stamp = (int(record["mtime_ns"]), int(record["size"]))
                memo[path_str] = (stamp, record["entry"])
            except (KeyError, TypeError, ValueError):
                continue
        return memo

    def _write_cache_file(self, memo: dict[str, tuple[_FileStamp, _CatalogEntry]]) -> None:
        path = self._cache_file()
        files = {
            path_str: {"mtime_ns": stamp[0], "size": stamp[1], "entry": entry}
            for path_str, (stamp, entry) in memo.items()
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"version": 2, "files": files}), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass  # cache is best-effort; a read-only root just means no speedup

    def _run_catalog_metadata(self, record: RunRecord) -> dict[str, Any]:
        from heimdall_run_view._forecaster import _forecaster_from_run_id, _normalize_forecaster_id

        summary = _read_run_summary_for_root(self.root, record)
        first = _read_first_jsonl(record.trace_path)
        setup_id = _setup_id(record.trace_path, record.run_id)
        return {
            "setup_id": setup_id,
            "setup_label": _setup_label(setup_id, record.run_id),
            "window_label": _window_label(record.run_id),
            "start_timestamp": _iso_z(first.get("timestamp") or first.get("utc_timestamp"))
            if first
            else None,
            "has_evaluation": bool(summary),
            "pnl_eur": _optional_float(summary.get("cumulative_pnl_eur")),
            "bid_action_count": _optional_int(summary.get("bid_action_count")),
            "cleared_mwh": _optional_float(summary.get("cleared_mwh")),
            "forecaster_id": _normalize_forecaster_id(
                first.get("forecaster_id") or _forecaster_from_run_id(record.run_id)
            ),
            "control_mode": _control_mode(record.run_id),
        }

    def get(self, run_id: str) -> RunRecord:
        direct = self._get_direct(run_id)
        if direct is not None:
            return direct
        for record in self._discover():
            if record.run_id == run_id:
                return record
        raise KeyError(run_id)

    def _get_direct(self, run_id: str) -> RunRecord | None:
        candidates = [
            self.root / "research" / "llm" / "ai-society" / "runs" / run_id / "traces.jsonl",
            *(self.root.glob(f"research/llm/ai-society/runs/*/{run_id}/traces.jsonl")),
        ]
        for trace_path in candidates:
            if not trace_path.exists():
                continue
            metadata = _scan_trace_metadata(trace_path)
            if metadata is None:
                continue
            first, total_steps = metadata
            actual_run_id = str(first.get("run_id") or trace_path.parent.name)
            if actual_run_id != run_id and trace_path.parent.name != run_id:
                continue
            manifest_path = self.root / "research" / "llm" / "evaluations" / actual_run_id / "manifest.json"
            if not manifest_path.exists():
                manifest_path = trace_path.parent / "manifest.json"
            return RunRecord(
                run_id=actual_run_id,
                trace_path=trace_path,
                manifest_path=manifest_path if manifest_path.exists() else None,
                trace_sha256=_file_sha256(trace_path),
                total_steps=total_steps,
            )
        return None

    def _record_for_trace(self, trace_path: Path) -> RunRecord | None:
        metadata = _scan_trace_metadata(trace_path)
        if metadata is None:
            return None
        first, total_steps = metadata
        run_id = str(first.get("run_id") or trace_path.parent.name)
        manifest_path = self.root / "research" / "llm" / "evaluations" / run_id / "manifest.json"
        if not manifest_path.exists():
            manifest_path = trace_path.parent / "manifest.json"
        return RunRecord(
            run_id=run_id,
            trace_path=trace_path,
            manifest_path=manifest_path if manifest_path.exists() else None,
            trace_sha256=_file_sha256(trace_path),
            total_steps=total_steps,
        )

    def _discover(self) -> list[RunRecord]:
        records: list[RunRecord] = []
        for trace_path in self.root.glob("research/llm/ai-society/runs/**/traces.jsonl"):
            record = self._record_for_trace(trace_path)
            if record is not None:
                records.append(record)
        return records


# ---------------------------------------------------------------------------
# Setup / window / control-mode helpers
# ---------------------------------------------------------------------------


def _setup_id(trace_path: Path, run_id: str) -> str:
    parent = trace_path.parent.parent.name
    if parent and parent != "runs":
        return parent
    if run_id.startswith("mixed20-"):
        return "mixed20"
    if run_id.startswith("msa-"):
        return "mixed-sideaware"
    if run_id.startswith("acm-"):
        return "action-core-matrix"
    if run_id.startswith("atm-"):
        return "adaptive-thesis-matrix"
    if run_id.startswith("rfm-"):
        return "risk-filter-matrix"
    if run_id.startswith("bfa-"):
        return "broadcast-forecaster-ablations"
    if run_id.startswith("icm-"):
        return "intelligence-chair-matrix"
    return run_id.split("-")[0] if "-" in run_id else "standalone"


def _setup_label(setup_id: str, run_id: str) -> str:
    labels = {
        "mixed20-full-days": "Mixed-20 full days",
        "mixed-sideaware-20260515": "Mixed-20 side-aware screens",
        "mixed-sideaware": "Mixed-20 side-aware screens",
        "action-core-matrix": "Action-core matrix",
        "adaptive-thesis-matrix": "Adaptive thesis matrix",
        "risk-filter-matrix": "Risk-filter matrix",
        "broadcast-forecaster-ablations": "Broadcast forecaster ablations",
        "intelligence-chair-matrix": "Intelligence chair matrix",
        "mixed20": "Mixed-20 runs",
    }
    if setup_id in labels:
        return labels[setup_id]
    return setup_id.replace("-", " ").title() if setup_id else run_id


def _window_label(run_id: str) -> str:
    tokens = run_id.split("-")
    pieces: list[str] = []
    for index, token in enumerate(tokens):
        if token.startswith("apr") and len(token) == 5:
            pieces.append(f"Apr {token[3:]}")
            if (
                index + 1 < len(tokens)
                and tokens[index + 1].isdigit()
                and len(tokens[index + 1]) == 4
            ):
                time = tokens[index + 1]
                pieces.append(f"{time[:2]}:{time[2:]}")
        elif token.isdigit() and int(token) in {2, 10, 12, 24, 48, 96}:
            pieces.append(f"{token} ticks")
        elif (
            token in {"real", "proxy"}
            and index + 1 < len(tokens)
            and tokens[index + 1] == "controls"
        ):
            pieces.append(f"{token} controls")
        elif token in {"q14", "q32", "q72"}:
            pieces.append(token.upper())
        elif token in {"f0", "f7", "f8", "f3_ensemble"}:
            pieces.append(token.upper())
    if pieces:
        deduped = list(dict.fromkeys(pieces))
        return " / ".join(deduped)
    return run_id


def _control_mode(run_id: str) -> str | None:
    lowered = run_id.lower()
    if "real-controls" in lowered:
        return "real controls"
    if "proxy-controls" in lowered:
        return "proxy controls"
    if "-det" in lowered or lowered.endswith("-det"):
        return "deterministic"
    if "-llm" in lowered or "-q" in lowered:
        return "llm"
    return None
