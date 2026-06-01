from datetime import UTC, datetime

import pytest

from packages.data import validate_window


def test_train_window_cannot_cross_structural_break() -> None:
    with pytest.raises(ValueError, match="2025-03-04"):
        validate_window(
            split="train",
            start=datetime(2025, 2, 1, tzinfo=UTC),
            end=datetime(2025, 3, 5, tzinfo=UTC),
        )


def test_post_break_window_can_cross_structural_break_when_explicit() -> None:
    validate_window(
        split="post_break",
        start=datetime(2025, 3, 4, tzinfo=UTC),
        end=datetime(2025, 3, 5, tzinfo=UTC),
    )
