"""Detect photo-majority folders so album images are skipped at walk time."""

from __future__ import annotations

from pathlib import Path

from infrarag.config import OcrConfig
from infrarag.ingest.ocr import IMAGE_EXTENSIONS

_photo_folder_cache: dict[str, bool] = {}


def clear_photo_folder_cache() -> None:
    """Reset cache (tests)."""
    _photo_folder_cache.clear()


def is_photo_folder(directory: Path, config: OcrConfig) -> bool:
    """True if directory looks like a photo album (majority image files)."""
    if not config.skip_photo_folders:
        return False
    key = str(directory.resolve())
    cached = _photo_folder_cache.get(key)
    if cached is not None:
        return cached

    file_count = 0
    image_count = 0
    try:
        for child in directory.iterdir():
            if not child.is_file():
                continue
            if not child.suffix:
                continue
            file_count += 1
            if child.suffix.lower() in IMAGE_EXTENSIONS:
                image_count += 1
    except OSError:
        _photo_folder_cache[key] = False
        return False

    result = (
        file_count >= config.photo_folder_min_files
        and (image_count / file_count) >= config.photo_folder_image_ratio
    )
    _photo_folder_cache[key] = result
    return result


def should_skip_image(path: Path, config: OcrConfig) -> bool:
    """Skip image files that live in photo-majority parent directories."""
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    return is_photo_folder(path.parent, config)
