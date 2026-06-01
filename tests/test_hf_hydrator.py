from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from heimdall_forecaster.inference import hf_hydrator


REQUIRED_FILES = ("config.json", "model.pt", "stats.pkl")


def test_checkpoint_dir_returns_ready_local_dir_without_download(tmp_path, monkeypatch) -> None:
    root = tmp_path / "models" / "forecaster"
    target = root / "f7" / "seed-42"
    target.mkdir(parents=True)
    for filename in REQUIRED_FILES:
        (target / filename).write_text("ok")

    monkeypatch.setattr(hf_hydrator, "DEFAULT_FORECASTER_ROOT", root)

    def fail_download(**_kwargs) -> None:
        raise AssertionError("snapshot_download should not be called")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fail_download),
    )

    assert hf_hydrator.checkpoint_dir("f7", 42, required_files=REQUIRED_FILES) == target


def test_checkpoint_dir_hydrates_when_partial_dir_is_missing_required_files(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "models" / "forecaster"
    repo_root = tmp_path / "repo"
    target = root / "f8" / "seed-42"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")

    monkeypatch.setattr(hf_hydrator, "DEFAULT_FORECASTER_ROOT", root)
    monkeypatch.setattr(hf_hydrator, "REPO_ROOT", repo_root)
    # Isolate from any HF_TOKEN in the ambient environment so the asserted
    # download kwargs are deterministic (the hydrator passes token=None when unset).
    monkeypatch.delenv("HF_TOKEN", raising=False)

    calls: list[dict[str, object]] = []

    def fake_download(**kwargs) -> None:
        calls.append(kwargs)
        for filename in REQUIRED_FILES:
            (target / filename).write_text("ok")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_download),
    )

    assert hf_hydrator.checkpoint_dir("f8", 42, required_files=REQUIRED_FILES) == target
    assert calls == [
        {
            "repo_id": hf_hydrator.HF_REPO,
            "allow_patterns": [
                "models/forecaster/f8/seed-42/*",
                "f8/seed-42/*",
            ],
            "local_dir": repo_root,
            "token": None,
        }
    ]


def test_checkpoint_dir_normalises_flat_hf_layout(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "models" / "forecaster"
    repo_root = tmp_path / "repo"
    flat_source = repo_root / "f11" / "seed-42"
    target = root / "f11" / "seed-42"

    monkeypatch.setattr(hf_hydrator, "DEFAULT_FORECASTER_ROOT", root)
    monkeypatch.setattr(hf_hydrator, "REPO_ROOT", repo_root)

    def fake_download(**_kwargs) -> None:
        flat_source.mkdir(parents=True, exist_ok=True)
        for filename in REQUIRED_FILES:
            (flat_source / filename).write_text("ok")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_download),
    )

    assert hf_hydrator.checkpoint_dir("f11", 42, required_files=REQUIRED_FILES) == target
    for filename in REQUIRED_FILES:
        assert (target / filename).read_text() == "ok"


def test_checkpoint_dir_reports_missing_required_files_after_hydration(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "models" / "forecaster"
    target = root / "f11" / "seed-42"

    monkeypatch.setattr(hf_hydrator, "DEFAULT_FORECASTER_ROOT", root)
    monkeypatch.setattr(hf_hydrator, "REPO_ROOT", tmp_path / "repo")

    def fake_download(**_kwargs) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_download),
    )

    with pytest.raises(FileNotFoundError, match="model.pt, stats.pkl"):
        hf_hydrator.checkpoint_dir("f11", 42, required_files=REQUIRED_FILES)
