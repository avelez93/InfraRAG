"""Tests for web threshold, profiles, and confirmed memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from infrarag.config import load_config
from infrarag.models import Chunk, ProposedFact, QueryResult
from infrarag.rag.memory import (
    dedupe_facts,
    format_memory_markdown,
    is_duplicate_fact,
    normalize_fact_text,
    save_confirmed_facts,
)
from infrarag.rag.profiles import (
    load_profile_context,
    resolve_memory_dir,
    resolve_org_dir,
    resolve_user_dir,
)
from infrarag.rag.retrieve import max_score, should_use_web

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def test_web_threshold_gate(default_config) -> None:
    cfg = default_config
    assert should_use_web(mode="deep", max_local_score=0.9, config=cfg) is True
    assert should_use_web(mode="ask", max_local_score=0.4, config=cfg) is True
    assert should_use_web(mode="ask", max_local_score=0.5, config=cfg) is True
    assert should_use_web(mode="ask", max_local_score=0.51, config=cfg) is False


def test_web_disabled(default_config, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    cfg = replace(default_config, web=replace(default_config.web, enabled=False))
    assert should_use_web(mode="deep", max_local_score=0.0, config=cfg) is False


def test_max_score_empty_and_hits() -> None:
    assert max_score([]) == 0.0
    hits = [
        QueryResult(
            chunk=Chunk(chunk_id="1", source_path="a.md", text="x"),
            score=0.2,
        ),
        QueryResult(
            chunk=Chunk(chunk_id="2", source_path="b.md", text="y"),
            score=0.7,
        ),
    ]
    assert max_score(hits) == 0.7


def test_profile_path_resolution(default_config) -> None:
    org = resolve_org_dir(default_config, base=REPO_ROOT)
    user = resolve_user_dir(default_config, base=REPO_ROOT)
    memory = resolve_memory_dir(default_config, base=REPO_ROOT)
    assert org == (REPO_ROOT / "profiles" / "org").resolve()
    assert user == (REPO_ROOT / "profiles" / "users" / "default").resolve()
    assert memory == user / "memory"
    assert (org / "profile.md").is_file()
    assert (user / "profile.md").is_file()


def test_load_profile_context_excludes_memory(tmp_path: Path, default_config) -> None:
    from dataclasses import replace

    org = tmp_path / "profiles" / "org"
    user = tmp_path / "profiles" / "users" / "default"
    memory = user / "memory"
    org.mkdir(parents=True)
    memory.mkdir(parents=True)
    (org / "profile.md").write_text("OrgName: Acme", encoding="utf-8")
    (user / "profile.md").write_text("User: Ada", encoding="utf-8")
    (memory / "old.md").write_text("Secret memory fact XYZ", encoding="utf-8")

    cfg = replace(
        default_config,
        profiles=replace(
            default_config.profiles,
            org_dir="profiles/org",
            users_root="profiles/users",
            user_id="default",
        ),
    )
    ctx = load_profile_context(cfg, base=tmp_path)
    assert "Acme" in ctx.org_text
    assert "Ada" in ctx.user_text
    assert "Secret memory fact XYZ" not in ctx.combined_text


def test_user_id_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFRARAG_USER_ID", "alice")
    config = load_config(DEFAULT_CONFIG)
    assert config.profiles.user_id == "alice"


def test_normalize_and_dedupe() -> None:
    known = "User files taxes in Navarra. Works at Acme Corp."
    fact_dup = ProposedFact(
        summary="Tax residency",
        text="User files taxes in Navarra.",
    )
    fact_new = ProposedFact(
        summary="Preferred editor",
        text="Prefers Neovim for infrastructure docs.",
    )
    assert is_duplicate_fact(fact_dup, known) is True
    assert is_duplicate_fact(fact_new, known) is False
    kept = dedupe_facts([fact_dup, fact_new, fact_new], known)
    assert len(kept) == 1
    assert kept[0].summary == "Preferred editor"
    assert normalize_fact_text("  Hello   World ") == "hello world"


def test_memory_markdown_write(tmp_path: Path, default_config) -> None:
    from dataclasses import replace

    user = tmp_path / "profiles" / "users" / "default"
    (user / "memory").mkdir(parents=True)
    (user / "profile.md").write_text("# User\n", encoding="utf-8")
    (tmp_path / "profiles" / "org").mkdir(parents=True)
    (tmp_path / "profiles" / "org" / "profile.md").write_text("# Org\n", encoding="utf-8")

    cfg = replace(
        default_config,
        profiles=replace(
            default_config.profiles,
            org_dir="profiles/org",
            users_root="profiles/users",
            user_id="default",
        ),
    )
    facts = [
        ProposedFact(summary="Role", text="Works as a platform engineer."),
    ]
    path, report = save_confirmed_facts(facts, cfg, base=tmp_path, ingest=False)
    assert path is not None
    assert path.is_file()
    assert report is None
    content = path.read_text(encoding="utf-8")
    assert "Confirmed memory" in content
    assert "Role" in content
    assert "platform engineer" in content
    assert "## Role" in format_memory_markdown(facts)

    # Second save of the same fact should be skipped after re-read
    path2, _ = save_confirmed_facts(facts, cfg, base=tmp_path, ingest=False)
    assert path2 is None
