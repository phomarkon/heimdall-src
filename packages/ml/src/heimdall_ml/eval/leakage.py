"""Pre/post-break leakage assertions for the test set.

Per docs/RESEARCH-PROPOSAL.md §5.7 the test window 2025-05-01 -> 2026-04-30 is
*sacred*: it is evaluated exactly once per final config. To make accidental
leakage hard, every training script's entry point calls ``assert_no_test_overlap``
before reading any panel.

The assertion is intentionally simple — we check that the timestamp range of
the supplied panel ends *before* the test-set start. We do **not** ship a way
to disable this check at runtime; reviewers should be able to grep for
``assert_no_test_overlap`` and convince themselves no script bypasses it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Per docs/RESEARCH-PROPOSAL.md §5.7 (frozen split)
TRAIN_END_UTC = datetime(2025, 3, 1, tzinfo=timezone.utc)  # train ≤ 2025-02-28
VAL_START_UTC = datetime(2025, 3, 4, tzinfo=timezone.utc)
VAL_END_UTC = datetime(2025, 4, 30, 23, 45, tzinfo=timezone.utc)
TEST_START_UTC = datetime(2025, 5, 1, tzinfo=timezone.utc)
TEST_END_UTC = datetime(2026, 4, 30, 23, 45, tzinfo=timezone.utc)


def assert_no_test_overlap(panel_path: Path | str, *, role: str) -> None:
    """Raise ValueError if the panel timestamp range overlaps the test window.

    ``role`` is one of {'train', 'val'} — purely diagnostic, written into the
    error message. Test-set evaluation must go through a separate code path
    (``experiments/test_eval.py``) which records the run ID in MLflow.
    """
    if role not in {"train", "val"}:
        raise ValueError(f"role must be 'train' or 'val', got {role!r}")
    p = Path(panel_path)
    df = pl.read_parquet(p, columns=["timestamp_utc"])
    if df.is_empty():
        raise ValueError(f"{role} panel {p} is empty")
    t_min = df["timestamp_utc"].min()
    t_max = df["timestamp_utc"].max()
    if t_max >= TEST_START_UTC:
        raise ValueError(
            f"LEAKAGE: {role} panel {p} ends at {t_max} >= test-set start "
            f"{TEST_START_UTC}. The test window is sacred (proposal §5.7)."
        )
    if role == "train" and t_max >= VAL_START_UTC:
        raise ValueError(
            f"LEAKAGE: train panel {p} ends at {t_max} >= val-set start "
            f"{VAL_START_UTC}; train must end at or before 2025-02-28."
        )
    if role == "val" and t_min < VAL_START_UTC:
        raise ValueError(
            f"LEAKAGE: val panel {p} starts at {t_min} < val-set start "
            f"{VAL_START_UTC}; pre-break data must not be in val."
        )


def assert_test_panel_only(panel_path: Path | str) -> None:
    """Use only inside ``experiments/test_eval.py``. Asserts the panel is
    exactly the test window."""
    p = Path(panel_path)
    df = pl.read_parquet(p, columns=["timestamp_utc"])
    t_min = df["timestamp_utc"].min()
    t_max = df["timestamp_utc"].max()
    if t_min < TEST_START_UTC or t_max > TEST_END_UTC:
        raise ValueError(
            f"test panel {p} range [{t_min}, {t_max}] outside the test window "
            f"[{TEST_START_UTC}, {TEST_END_UTC}]"
        )


__all__ = [
    "TEST_END_UTC",
    "TEST_START_UTC",
    "TRAIN_END_UTC",
    "VAL_END_UTC",
    "VAL_START_UTC",
    "assert_no_test_overlap",
    "assert_test_panel_only",
]
