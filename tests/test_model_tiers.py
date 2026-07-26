"""Tests for model tier selection."""

from __future__ import annotations

import pytest

from infrarag.model_tiers import get_tier, select_tier


@pytest.mark.parametrize(
    ("effective_gb", "expected_id"),
    [
        (0.0, "nano"),
        (3.9, "nano"),
        (4.0, "tiny"),
        (5.9, "tiny"),
        (6.0, "compact"),
        (8.9, "compact"),
        (9.0, "small"),
        (13.9, "small"),
        (14.0, "medium"),
        (23.9, "medium"),
        (24.0, "large"),
        (39.9, "large"),
        (40.0, "xlarge"),
        (100.0, "xlarge"),
    ],
)
def test_select_tier(effective_gb: float, expected_id: str) -> None:
    assert select_tier(effective_gb).id == expected_id


def test_get_tier() -> None:
    assert get_tier("COMPACT").chat_model == "qwen2.5:3b"
    with pytest.raises(KeyError):
        get_tier("nope")
