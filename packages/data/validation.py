from __future__ import annotations

from datetime import UTC, datetime


STRUCTURAL_BREAK_UTC = datetime(2025, 3, 4, tzinfo=UTC)


def validate_window(*, split: str, start: datetime, end: datetime) -> None:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Data windows must be timezone-aware")
    if start >= end:
        raise ValueError("Data window start must be before end")
    if split == "train" and end > STRUCTURAL_BREAK_UTC:
        raise ValueError(
            "train windows cannot cross the 2025-03-04 structural break"
        )
