"""Differential ingest fingerprints and manifest comparison (stub)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileFingerprint:
    """Identity of a source file for differential ingest."""

    relative_path: str
    mtime_ns: int
    size: int


def fingerprint_file(path: Path, *, root: Path) -> FileFingerprint:
    """Build a fingerprint from relative path, mtime_ns, and size."""
    raise NotImplementedError("fingerprint_file is not implemented yet")


def load_manifest(manifest_path: Path) -> dict[str, FileFingerprint]:
    """Load the ingest manifest mapping relative paths to fingerprints."""
    raise NotImplementedError("load_manifest is not implemented yet")


def save_manifest(manifest_path: Path, entries: dict[str, FileFingerprint]) -> None:
    """Persist the ingest manifest to disk."""
    raise NotImplementedError("save_manifest is not implemented yet")


def classify_change(
    current: FileFingerprint,
    previous: FileFingerprint | None,
) -> str:
    """Return 'added', 'updated', or 'unchanged' for a fingerprint pair."""
    raise NotImplementedError("classify_change is not implemented yet")
