"""Load always-include org and user profile context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from infrarag.config import AppConfig, ProfilesConfig
from infrarag.ingest.loaders import load_document


@dataclass(frozen=True)
class ProfileContext:
    """Text from org + user profile files (memory/ excluded)."""

    org_text: str
    user_text: str
    org_sources: tuple[str, ...]
    user_sources: tuple[str, ...]

    @property
    def combined_text(self) -> str:
        parts: list[str] = []
        if self.org_text.strip():
            parts.append(f"[profile:org]\n{self.org_text.strip()}")
        if self.user_text.strip():
            parts.append(f"[profile:user]\n{self.user_text.strip()}")
        return "\n\n".join(parts)


def resolve_org_dir(config: AppConfig | ProfilesConfig, *, base: Path | None = None) -> Path:
    """Resolve the organization profile directory."""
    profiles = config if isinstance(config, ProfilesConfig) else config.profiles
    root = base or Path.cwd()
    return (root / profiles.org_dir).expanduser().resolve()


def resolve_user_dir(config: AppConfig | ProfilesConfig, *, base: Path | None = None) -> Path:
    """Resolve the active user profile directory."""
    profiles = config if isinstance(config, ProfilesConfig) else config.profiles
    root = base or Path.cwd()
    return (root / profiles.users_root / profiles.user_id).expanduser().resolve()


def resolve_memory_dir(config: AppConfig | ProfilesConfig, *, base: Path | None = None) -> Path:
    """Resolve the confirmed-memory directory for the active user."""
    return resolve_user_dir(config, base=base) / "memory"


def _profile_extensions(config: AppConfig) -> set[str]:
    return {ext.lower() for ext in config.ingest.include_extensions}


def _iter_profile_files(directory: Path, *, extensions: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        # Exclude memory/ from always-include (indexed separately after confirm).
        try:
            relative = path.relative_to(directory)
        except ValueError:
            continue
        if "memory" in relative.parts:
            continue
        if path.suffix.lower() not in extensions:
            continue
        files.append(path)
    return files


def _load_dir_text(
    directory: Path,
    config: AppConfig,
    *,
    max_chars: int,
) -> tuple[str, tuple[str, ...]]:
    extensions = _profile_extensions(config)
    parts: list[str] = []
    sources: list[str] = []
    remaining = max_chars
    for path in _iter_profile_files(directory, extensions=extensions):
        if remaining <= 0:
            break
        try:
            doc = load_document(path, ocr=config.ingest.ocr)
        except Exception:  # noqa: BLE001
            continue
        text = doc.text.strip()
        if not text:
            continue
        if len(text) > remaining:
            text = text[:remaining]
        parts.append(f"### {path.name}\n{text}")
        sources.append(str(path))
        remaining -= len(text)
    return "\n\n".join(parts), tuple(sources)


def load_profile_context(config: AppConfig, *, base: Path | None = None) -> ProfileContext:
    """Load org + user profile files, truncated to always_include_max_chars total."""
    max_chars = config.profiles.always_include_max_chars
    org_dir = resolve_org_dir(config, base=base)
    user_dir = resolve_user_dir(config, base=base)
    # Split budget roughly evenly; unused org budget falls through unused.
    org_budget = max_chars // 2
    org_text, org_sources = _load_dir_text(org_dir, config, max_chars=org_budget)
    user_budget = max_chars - len(org_text)
    user_text, user_sources = _load_dir_text(user_dir, config, max_chars=max(0, user_budget))
    return ProfileContext(
        org_text=org_text,
        user_text=user_text,
        org_sources=org_sources,
        user_sources=user_sources,
    )


def load_known_memory_text(config: AppConfig, *, base: Path | None = None) -> str:
    """Load all markdown (and other ingestable) files under the user memory folder."""
    memory_dir = resolve_memory_dir(config, base=base)
    if not memory_dir.is_dir():
        return ""
    extensions = _profile_extensions(config)
    parts: list[str] = []
    for path in sorted(memory_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.name == ".gitkeep":
            continue
        try:
            doc = load_document(path, ocr=config.ingest.ocr)
        except Exception:  # noqa: BLE001
            continue
        text = doc.text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)
