"""Tests for ingest root tracking and inference."""

from __future__ import annotations

from pathlib import Path

from infrarag.ingest.roots import (
    infer_roots_from_source_paths,
    load_ingest_roots,
    record_ingest_root,
    save_ingest_roots,
)


def test_record_and_load_ingest_roots(tmp_path: Path, default_config) -> None:
    from dataclasses import replace

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    cfg = replace(
        default_config,
        chroma=replace(default_config.chroma, persist_dir=str(chroma_dir)),
    )
    a = tmp_path / "docs_a"
    b = tmp_path / "docs_b"
    a.mkdir()
    b.mkdir()
    record_ingest_root(cfg, a)
    record_ingest_root(cfg, b)
    record_ingest_root(cfg, a)  # dedupe
    roots = load_ingest_roots(cfg)
    assert len(roots) == 2
    assert a.resolve() in roots
    assert b.resolve() in roots


def test_infer_roots_promotes_year_siblings(tmp_path: Path) -> None:
    base = tmp_path / "informes"
    y2023 = base / "2023"
    y2024 = base / "2024"
    y2023.mkdir(parents=True)
    y2024.mkdir(parents=True)
    f1 = y2023 / "a.pdf"
    f2 = y2024 / "b.pdf"
    f1.write_text("x", encoding="utf-8")
    f2.write_text("y", encoding="utf-8")
    roots = infer_roots_from_source_paths([str(f1), str(f2)])
    assert any(r.resolve() == base.resolve() for r in roots)


def test_save_ingest_roots_roundtrip(tmp_path: Path, default_config) -> None:
    from dataclasses import replace

    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    cfg = replace(
        default_config,
        chroma=replace(default_config.chroma, persist_dir=str(chroma_dir)),
    )
    root = tmp_path / "corpus"
    root.mkdir()
    save_ingest_roots(cfg, [root])
    loaded = load_ingest_roots(cfg)
    assert loaded == [root.resolve()]
