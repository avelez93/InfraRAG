"""Walk a source directory and yield readable file paths (stub)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from infrarag.config import IngestConfig


def iter_source_files(root: Path, config: IngestConfig) -> Iterator[Path]:
    """Recursively yield files under root that match include/exclude rules.

    Skips excluded directory names, oversized files, and unsupported extensions.
    """
    raise NotImplementedError("iter_source_files is not implemented yet")
