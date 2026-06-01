"""Smoke tests for the canonical apples-to-apples runner.

Each test runs one model family on a tiny pre-carved slice with the minimum
budget that still exercises the full code path (data load -> windowing ->
fit -> val pinball -> persistence). The goal is to validate the pipeline,
not the quality of any single model. Real experiments run from
``train.canonical`` CLI on the full panels.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from heimdall_forecaster.train import canonical as C
from heimdall_forecaster.train.dataset import (
    CANONICAL_FEATURE_GROUPS,
    F_CANONICAL_FEATURES,
    f_canonical_without,
)


def test_logo_groups_subset_of_canonical() -> None:
    """Every LOGO group column must exist in F_CANONICAL — guards against typos."""
    canon = set(F_CANONICAL_FEATURES)
    for group, cols in CANONICAL_FEATURE_GROUPS.items():
        missing = [c for c in cols if c not in canon]
        assert not missing, f"group {group!r} references non-canonical cols {missing}"
        rest = f_canonical_without(group)
        assert len(rest) == len(F_CANONICAL_FEATURES) - len(set(cols) & canon)

SMOKE = Path("/tmp/heimdall_smoke")
pytestmark = pytest.mark.skipif(
    not (SMOKE / "train.parquet").exists(),
    reason="smoke fixtures not carved; see test docstring",
)


def _patch_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(C, "DEFAULT_TRAIN", SMOKE / "train.parquet")
    monkeypatch.setattr(C, "DEFAULT_VAL", SMOKE / "val.parquet")
    monkeypatch.setattr(C, "DEFAULT_ANOMALY", SMOKE / "anom.parquet")
    monkeypatch.setattr(C, "UNIVARIATE_TRAIN", SMOKE / "uni_train.parquet")
    monkeypatch.setattr(C, "UNIVARIATE_VAL", SMOKE / "uni_val.parquet")


def _assert_finite_metrics(row: dict) -> None:
    pinball = row.get("val_pinball_mean")
    assert pinball is not None, row
    assert math.isfinite(pinball), row


@pytest.mark.parametrize("target", ["price", "activation"])
def test_canonical_f8_multivariate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   target: str) -> None:
    _patch_paths(monkeypatch)
    row = C.run_one("f8", target, seed=42, smoke=True, out_dir=tmp_path)
    assert row["routing"] == "multivariate", row
    _assert_finite_metrics(row)


@pytest.mark.parametrize("target", ["price", "activation"])
def test_canonical_f0_univariate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 target: str) -> None:
    _patch_paths(monkeypatch)
    row = C.run_one("f0", target, seed=42, smoke=True, out_dir=tmp_path)
    assert row["routing"] == "univariate", row
    _assert_finite_metrics(row)
