"""Tests for differential fingerprints and chunking / walker."""

from __future__ import annotations

from pathlib import Path

import pytest

from infrarag.config import ChunkingConfig, IngestConfig, OcrConfig, load_config
from infrarag.ingest.chunking import chunk_document
from infrarag.ingest.differential import (
    FileFingerprint,
    classify_change,
    fingerprint_file,
    load_manifest,
    save_manifest,
)
from infrarag.ingest.loaders import load_document
from infrarag.ingest.walker import iter_source_files
from infrarag.models import Document


def _default_ocr() -> OcrConfig:
    return OcrConfig(
        enabled=True,
        lang="es",
        use_gpu=False,
        min_text_chars=40,
        max_pdf_pages=50,
        skip_photo_folders=True,
        photo_folder_image_ratio=0.6,
        photo_folder_min_files=5,
    )


def test_classify_change() -> None:
    cur = FileFingerprint("a.txt", 1, 10)
    assert classify_change(cur, None) == "added"
    assert classify_change(cur, cur) == "unchanged"
    assert classify_change(cur, FileFingerprint("a.txt", 2, 10)) == "updated"


def test_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    entries = {"a.txt": FileFingerprint("a.txt", 123, 5)}
    save_manifest(path, entries)
    loaded = load_manifest(path)
    assert loaded["a.txt"].mtime_ns == 123
    assert loaded["a.txt"].size == 5


def test_fingerprint_and_walker(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "note.md").write_text("# hi", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.txt").write_text("no", encoding="utf-8")

    config = IngestConfig(
        source_dir=str(tmp_path),
        recursive=True,
        follow_symlinks=False,
        differential=True,
        manifest_path=str(tmp_path / "m.json"),
        include_extensions=(".txt", ".md"),
        exclude_dirs=(".git",),
        max_file_bytes=1_000_000,
        ocr=_default_ocr(),
    )
    files = sorted(iter_source_files(tmp_path, config))
    names = {p.name for p in files}
    assert names == {"keep.txt", "note.md"}

    fp = fingerprint_file(tmp_path / "keep.txt", root=tmp_path)
    assert fp.relative_path == "keep.txt"
    assert fp.size == 5


def test_chunk_document() -> None:
    doc = Document(source_path="/tmp/a.txt", text="abcdefghij")
    chunks = chunk_document(doc, ChunkingConfig(size=4, overlap=1))
    assert [c.text for c in chunks] == ["abcd", "defg", "ghij"]
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_load_txt_md(tmp_path: Path) -> None:
    p = tmp_path / "a.md"
    p.write_text("# Title\nbody", encoding="utf-8")
    doc = load_document(p)
    assert "Title" in doc.text


def test_local_yaml_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import infrarag.config as cfg

    default = tmp_path / "default.yaml"
    local = tmp_path / "local.yaml"
    default.write_text(
        """
app:
  name: InfraRAG
  data_dir: data
ollama:
  base_url: http://localhost:11434
  chat_model: qwen2.5:3b
  embed_model: nomic-embed-text
  timeout_s: 120
chroma:
  persist_dir: data/chroma
  collection: infrarag
ingest:
  source_dir: null
  recursive: true
  follow_symlinks: false
  differential: true
  manifest_path: data/chroma/ingest_manifest.json
  include_extensions: [.txt]
  exclude_dirs: [.git]
  max_file_bytes: 1000
  ocr:
    enabled: true
    lang: es
    use_gpu: false
    min_text_chars: 40
    max_pdf_pages: 50
    skip_photo_folders: true
    photo_folder_image_ratio: 0.6
    photo_folder_min_files: 5
chunking:
  size: 800
  overlap: 120
rag:
  top_k: 5
  temperature: 0.2
attachments:
  max_file_bytes: 1000
  url_timeout_s: 30
  allowed_url_schemes: [http, https]
ui:
  host: 127.0.0.1
  port: 8501
""",
        encoding="utf-8",
    )
    local.write_text(
        "ollama:\n  chat_model: qwen3:8b\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "_DEFAULT_CONFIG_PATH", default)
    monkeypatch.setattr(cfg, "_LOCAL_CONFIG_PATH", local)
    config = load_config()
    assert config.ollama.chat_model == "qwen3:8b"
    assert config.ollama.embed_model == "nomic-embed-text"
