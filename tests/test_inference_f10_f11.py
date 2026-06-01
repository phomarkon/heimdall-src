"""Tests for the F10 (Chronos-Bolt) and F11 (PriceFM) appendix backends."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from heimdall_forecaster.inference import Forecaster, get_forecaster, list_registered


def test_f10_f11_in_registry() -> None:
    backends = list_registered()
    assert "f10" in backends
    assert "f11" in backends


@pytest.mark.slow
def test_f10_chronos_bolt_predicts(monkeypatch) -> None:
    pytest.importorskip("chronos", reason="F10 requires optional chronos-forecasting")
    monkeypatch.setenv("HEIMDALL_CHRONOS_MODEL", "amazon/chronos-bolt-tiny")
    # Clear the cache so the env override takes effect.
    from heimdall_forecaster.inference import clear_cache
    clear_cache()
    f = get_forecaster("f10")
    assert isinstance(f, Forecaster)
    qs = f.predict([100.0 + 0.1 * i for i in range(96)], horizon=4)
    assert len(qs) == 4
    for q in qs:
        assert len(q.values) == 3
        assert q.values[0] <= q.values[1] + 1e-3
        assert q.values[1] <= q.values[2] + 1e-3


def test_f11_raises_clearly_when_weights_absent(tmp_path, monkeypatch) -> None:
    """Without an F11 checkpoint dir, the loader fails fast with a recipe."""
    from heimdall_forecaster.inference import clear_cache
    from heimdall_forecaster.inference import hf_hydrator

    def fail_download(**_kwargs) -> None:
        raise FileNotFoundError("fixture blocks HF")

    monkeypatch.setattr(
        hf_hydrator,
        "DEFAULT_FORECASTER_ROOT",
        tmp_path / "models" / "forecaster",
    )
    monkeypatch.setattr(hf_hydrator, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fail_download),
    )

    clear_cache()
    with pytest.raises(FileNotFoundError) as exc_info:
        get_forecaster("f11")
    msg = str(exc_info.value)
    assert "PriceFM" in msg
    assert "MODEL_CARD" in msg or "fine-tune" in msg
