"""Pytest fixtures for InfraRAG."""

from __future__ import annotations

from pathlib import Path

import pytest

from infrarag.config import AppConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def default_config() -> AppConfig:
    return load_config(REPO_ROOT / "config" / "default.yaml")
