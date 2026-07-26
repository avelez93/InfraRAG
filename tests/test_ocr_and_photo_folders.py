"""Tests for OCR helpers and photo-folder skipping."""

from __future__ import annotations

from pathlib import Path

import pytest

from infrarag.config import OcrConfig, load_config
from infrarag.ingest.ocr import pdf_text_needs_ocr
from infrarag.ingest.photo_folders import (
    clear_photo_folder_cache,
    is_photo_folder,
    should_skip_image,
)
from infrarag.ingest.walker import iter_source_files
from infrarag.config import IngestConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ocr(**overrides: object) -> OcrConfig:
    base = dict(
        enabled=True,
        lang="es",
        use_gpu=False,
        min_text_chars=40,
        max_pdf_pages=50,
        skip_photo_folders=True,
        photo_folder_image_ratio=0.6,
        photo_folder_min_files=5,
    )
    base.update(overrides)
    return OcrConfig(**base)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_photo_folder_cache()


def test_pdf_text_needs_ocr() -> None:
    assert pdf_text_needs_ocr("", page_count=1, min_text_chars=40) is True
    assert pdf_text_needs_ocr("x" * 10, page_count=1, min_text_chars=40) is True
    # Long enough total but thin per page
    assert pdf_text_needs_ocr("x" * 50, page_count=10, min_text_chars=40) is True
    assert pdf_text_needs_ocr("word " * 40, page_count=1, min_text_chars=40) is False


def test_photo_folder_majority(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"p{i}.jpg").write_bytes(b"fake")
    assert is_photo_folder(tmp_path, _ocr()) is True
    assert should_skip_image(tmp_path / "p0.jpg", _ocr()) is True


def test_mixed_folder_keeps_images(tmp_path: Path) -> None:
    for i in range(2):
        (tmp_path / f"p{i}.png").write_bytes(b"fake")
    for i in range(8):
        (tmp_path / f"d{i}.txt").write_text("doc", encoding="utf-8")
    assert is_photo_folder(tmp_path, _ocr()) is False
    assert should_skip_image(tmp_path / "p0.png", _ocr()) is False


def test_small_folder_does_not_trigger_rule(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    assert is_photo_folder(tmp_path, _ocr()) is False


def test_walker_skips_album_images(tmp_path: Path) -> None:
    album = tmp_path / "album"
    album.mkdir()
    for i in range(5):
        (album / f"shot{i}.jpg").write_bytes(b"img")
    (album / "notes.txt").write_text("keep me", encoding="utf-8")

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "diagram.png").write_bytes(b"img")
    (docs / "readme.md").write_text("# hi", encoding="utf-8")

    config = IngestConfig(
        source_dir=str(tmp_path),
        recursive=True,
        follow_symlinks=False,
        differential=True,
        manifest_path=str(tmp_path / "m.json"),
        include_extensions=(".txt", ".md", ".png", ".jpg", ".jpeg"),
        exclude_dirs=(".git",),
        max_file_bytes=1_000_000,
        ocr=_ocr(),
    )
    names = {p.name for p in iter_source_files(tmp_path, config)}
    assert "notes.txt" in names
    assert "readme.md" in names
    assert "diagram.png" in names
    assert "shot0.jpg" not in names


def test_default_config_includes_ocr_and_images() -> None:
    config = load_config(REPO_ROOT / "config" / "default.yaml")
    assert ".png" in config.ingest.include_extensions
    assert ".jpg" in config.ingest.include_extensions
    assert config.ingest.ocr.enabled is True
    assert config.ingest.ocr.lang == "es"
    assert config.ingest.ocr.photo_folder_image_ratio == 0.6
