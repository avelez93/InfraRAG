"""Directory ingest orchestration (stub)."""

from __future__ import annotations

from pathlib import Path

from infrarag.config import AppConfig
from infrarag.models import IngestReport


def ingest_directory(path: Path, config: AppConfig) -> IngestReport:
    """Walk path, load readable files, differentially chunk and index into Chroma.

    When config.ingest.differential is True, only created or modified files
    (by fingerprint) are re-indexed.
    """
    raise NotImplementedError("ingest_directory is not implemented yet")
