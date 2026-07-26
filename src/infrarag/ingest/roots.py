"""Persist directories the user has indexed (for full re-index)."""

from __future__ import annotations

import json
from pathlib import Path

from infrarag.config import AppConfig


def _roots_path(config: AppConfig) -> Path:
    persist = Path(config.chroma.persist_dir)
    if not persist.is_absolute():
        persist = Path.cwd() / persist
    return persist / "ingest_roots.json"


def load_ingest_roots(config: AppConfig) -> list[Path]:
    """Return absolute directory paths previously used for ingest."""
    path = _roots_path(config)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    roots_raw = raw.get("roots") if isinstance(raw, dict) else None
    if not isinstance(roots_raw, list):
        return []
    out: list[Path] = []
    seen: set[str] = set()
    for item in roots_raw:
        try:
            resolved = str(Path(str(item)).expanduser().resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(Path(resolved))
    return out


def save_ingest_roots(config: AppConfig, roots: list[Path]) -> Path:
    """Write the ingest-roots list (absolute paths, sorted)."""
    path = _roots_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    unique: list[str] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.expanduser().resolve())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    unique.sort()
    payload = {"roots": unique}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def record_ingest_root(config: AppConfig, root: Path) -> None:
    """Append a directory to the known ingest roots if not already present."""
    roots = load_ingest_roots(config)
    try:
        resolved = root.expanduser().resolve()
    except OSError:
        return
    if resolved not in roots:
        roots.append(resolved)
        save_ingest_roots(config, roots)


def infer_roots_from_source_paths(source_paths: list[str]) -> list[Path]:
    """Best-effort roots when ingest_roots.json is empty: common parent per path cluster."""
    files = []
    for raw in source_paths:
        try:
            p = Path(raw).expanduser().resolve()
        except OSError:
            continue
        if p.is_file() or p.parent.is_dir():
            files.append(p)
    if not files:
        return []
    # Prefer parent directories of indexed files; keep only maximal roots (not under another).
    parents = sorted({f.parent for f in files}, key=lambda p: len(p.parts))
    roots: list[Path] = []
    for parent in parents:
        under_existing = False
        for root in list(roots):
            try:
                parent.relative_to(root)
                under_existing = True
                break
            except ValueError:
                pass
            try:
                root.relative_to(parent)
                # parent is above root — promote
                roots.remove(root)
            except ValueError:
                pass
        if not under_existing:
            roots.append(parent)
    # Promote one level when many sibling year folders share a grandparent
    # e.g. informes/2023 + informes/2024 -> prefer informes if both are roots.
    promoted: list[Path] = []
    by_grand: dict[Path, list[Path]] = {}
    for root in roots:
        by_grand.setdefault(root.parent, []).append(root)
    for grand, children in by_grand.items():
        if len(children) >= 2 and grand != Path(grand.anchor):
            promoted.append(grand)
        else:
            promoted.extend(children)
    # Deduplicate
    final: list[Path] = []
    seen: set[str] = set()
    for root in sorted(promoted, key=lambda p: len(p.parts)):
        key = str(root)
        if key in seen:
            continue
        if any(
            key != str(other) and key.startswith(str(other) + "/")
            for other in seen
        ):
            continue
        seen.add(key)
        final.append(root)
    return final
