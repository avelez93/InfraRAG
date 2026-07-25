"""Load and resolve InfraRAG configuration from YAML and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root: src/infrarag/config.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class AppSection:
    name: str
    data_dir: str


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    chat_model: str
    embed_model: str
    timeout_s: int


@dataclass(frozen=True)
class ChromaConfig:
    persist_dir: str
    collection: str


@dataclass(frozen=True)
class IngestConfig:
    source_dir: str | None
    recursive: bool
    follow_symlinks: bool
    differential: bool
    manifest_path: str
    include_extensions: tuple[str, ...]
    exclude_dirs: tuple[str, ...]
    max_file_bytes: int


@dataclass(frozen=True)
class ChunkingConfig:
    size: int
    overlap: int


@dataclass(frozen=True)
class RagConfig:
    top_k: int
    temperature: float


@dataclass(frozen=True)
class AttachmentsConfig:
    max_file_bytes: int
    url_timeout_s: int
    allowed_url_schemes: tuple[str, ...]


@dataclass(frozen=True)
class UiConfig:
    host: str
    port: int


@dataclass(frozen=True)
class AppConfig:
    app: AppSection
    ollama: OllamaConfig
    chroma: ChromaConfig
    ingest: IngestConfig
    chunking: ChunkingConfig
    rag: RagConfig
    attachments: AttachmentsConfig
    ui: UiConfig
    config_path: Path = field(repr=False)


def _as_tuple_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(item) for item in value)


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply INFRARAG_* environment overrides onto a loaded YAML dict."""
    data = {
        "app": dict(raw.get("app") or {}),
        "ollama": dict(raw.get("ollama") or {}),
        "chroma": dict(raw.get("chroma") or {}),
        "ingest": dict(raw.get("ingest") or {}),
        "chunking": dict(raw.get("chunking") or {}),
        "rag": dict(raw.get("rag") or {}),
        "attachments": dict(raw.get("attachments") or {}),
        "ui": dict(raw.get("ui") or {}),
    }

    if (v := _env("INFRARAG_OLLAMA_BASE_URL")) is not None:
        data["ollama"]["base_url"] = v
    if (v := _env("INFRARAG_CHAT_MODEL")) is not None:
        data["ollama"]["chat_model"] = v
    if (v := _env("INFRARAG_EMBED_MODEL")) is not None:
        data["ollama"]["embed_model"] = v
    if (v := _env("INFRARAG_CHROMA_DIR")) is not None:
        data["chroma"]["persist_dir"] = v
    if (v := _env("INFRARAG_SOURCE_DIR")) is not None:
        data["ingest"]["source_dir"] = v

    return data


def _build_config(data: dict[str, Any], config_path: Path) -> AppConfig:
    app = data["app"]
    ollama = data["ollama"]
    chroma = data["chroma"]
    ingest = data["ingest"]
    chunking = data["chunking"]
    rag = data["rag"]
    attachments = data["attachments"]
    ui = data["ui"]

    source_dir = ingest.get("source_dir")
    if source_dir is not None:
        source_dir = str(source_dir)

    return AppConfig(
        app=AppSection(name=str(app["name"]), data_dir=str(app["data_dir"])),
        ollama=OllamaConfig(
            base_url=str(ollama["base_url"]),
            chat_model=str(ollama["chat_model"]),
            embed_model=str(ollama["embed_model"]),
            timeout_s=int(ollama["timeout_s"]),
        ),
        chroma=ChromaConfig(
            persist_dir=str(chroma["persist_dir"]),
            collection=str(chroma["collection"]),
        ),
        ingest=IngestConfig(
            source_dir=source_dir,
            recursive=bool(ingest["recursive"]),
            follow_symlinks=bool(ingest["follow_symlinks"]),
            differential=bool(ingest["differential"]),
            manifest_path=str(ingest["manifest_path"]),
            include_extensions=_as_tuple_str(ingest.get("include_extensions")),
            exclude_dirs=_as_tuple_str(ingest.get("exclude_dirs")),
            max_file_bytes=int(ingest["max_file_bytes"]),
        ),
        chunking=ChunkingConfig(
            size=int(chunking["size"]),
            overlap=int(chunking["overlap"]),
        ),
        rag=RagConfig(
            top_k=int(rag["top_k"]),
            temperature=float(rag["temperature"]),
        ),
        attachments=AttachmentsConfig(
            max_file_bytes=int(attachments["max_file_bytes"]),
            url_timeout_s=int(attachments["url_timeout_s"]),
            allowed_url_schemes=_as_tuple_str(attachments.get("allowed_url_schemes")),
        ),
        ui=UiConfig(host=str(ui["host"]), port=int(ui["port"])),
        config_path=config_path,
    )


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load YAML config, apply env overrides, and return a frozen AppConfig."""
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    data = _apply_env_overrides(raw)
    return _build_config(data, config_path.resolve())
