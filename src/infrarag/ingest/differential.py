"""Differential ingest fingerprints and manifest comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileFingerprint:
    """Identity of a source file for differential ingest."""

    relative_path: str
    mtime_ns: int
    size: int


def fingerprint_file(path: Path, *, root: Path) -> FileFingerprint:
    """Build a fingerprint from relative path, mtime_ns, and size."""
    path = path.resolve()
    root = root.resolve()
    rel = str(path.relative_to(root)).replace("\\", "/")
    stat = path.stat()
    return FileFingerprint(relative_path=rel, mtime_ns=int(stat.st_mtime_ns), size=int(stat.st_size))


def load_manifest(manifest_path: Path) -> dict[str, FileFingerprint]:
    """Load the ingest manifest mapping relative paths to fingerprints."""
    if not manifest_path.is_file():
        return {}
    with manifest_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    entries: dict[str, FileFingerprint] = {}
    for key, value in (raw.get("files") or {}).items():
        entries[key] = FileFingerprint(
            relative_path=str(value["relative_path"]),
            mtime_ns=int(value["mtime_ns"]),
            size=int(value["size"]),
        )
    return entries


def save_manifest(manifest_path: Path, entries: dict[str, FileFingerprint]) -> None:
    """Persist the ingest manifest to disk."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "files": {key: asdict(fp) for key, fp in sorted(entries.items())},
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def classify_change(
    current: FileFingerprint,
    previous: FileFingerprint | None,
) -> str:
    """Return 'added', 'updated', or 'unchanged' for a fingerprint pair."""
    if previous is None:
        return "added"
    if (
        previous.mtime_ns == current.mtime_ns
        and previous.size == current.size
        and previous.relative_path == current.relative_path
    ):
        return "unchanged"
    return "updated"
