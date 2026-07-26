"""Walk a source directory and yield readable file paths."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from infrarag.config import IngestConfig
from infrarag.ingest.photo_folders import should_skip_image


def iter_source_files(root: Path, config: IngestConfig) -> Iterator[Path]:
    """Recursively yield files under root that match include/exclude rules."""
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {root}")

    include = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in config.include_extensions
    }
    exclude = set(config.exclude_dirs)

    if config.recursive:
        walker = root.rglob("*")
    else:
        walker = root.glob("*")

    for path in walker:
        if not path.is_file():
            continue
        if any(part in exclude for part in path.parts):
            continue
        if not config.follow_symlinks and path.is_symlink():
            continue
        suffix = path.suffix.lower()
        if suffix not in include:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > config.max_file_bytes:
            continue
        if should_skip_image(path, config.ocr):
            continue
        yield path
