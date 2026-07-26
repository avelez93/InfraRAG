"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from infrarag.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def test_load_default_config() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.app.name == "InfraRAG"
    assert config.ollama.chat_model == "qwen2.5:3b"
    assert config.ollama.embed_model == "nomic-embed-text"
    assert config.chroma.collection == "infrarag"
    assert config.ingest.differential is True
    assert config.ingest.source_dir is None
    assert ".pdf" in config.ingest.include_extensions
    assert ".docx" in config.ingest.include_extensions
    assert ".odt" in config.ingest.include_extensions
    assert ".png" in config.ingest.include_extensions
    assert config.ingest.ocr.enabled is True
    assert config.ingest.ocr.min_text_chars == 40
    assert config.chunking.size == 800
    assert config.chunking.overlap == 120
    assert config.rag.top_k == 10
    assert config.rag.hybrid_enabled is True
    assert config.rag.hybrid_fetch_multiplier == 4
    assert config.rag.hybrid_keyword_weight == 0.35
    assert config.profiles.user_id == "default"
    assert config.profiles.org_dir == "profiles/org"
    assert config.web.enabled is True
    assert config.web.score_threshold == 0.5
    assert config.memory.enabled is True
    assert "http" in config.attachments.allowed_url_schemes
    assert config.ui.port == 8501


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFRARAG_CHAT_MODEL", "phi3:mini")
    monkeypatch.setenv("INFRARAG_SOURCE_DIR", "/tmp/docs")
    monkeypatch.setenv("INFRARAG_CHROMA_DIR", "/tmp/chroma")
    monkeypatch.setenv("INFRARAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("INFRARAG_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("INFRARAG_USER_ID", "bob")

    config = load_config(DEFAULT_CONFIG)

    assert config.ollama.chat_model == "phi3:mini"
    assert config.ollama.base_url == "http://127.0.0.1:11434"
    assert config.ollama.embed_model == "nomic-embed-text"
    assert config.ingest.source_dir == "/tmp/docs"
    assert config.chroma.persist_dir == "/tmp/chroma"
    assert config.profiles.user_id == "bob"


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/infrarag-config.yaml")
