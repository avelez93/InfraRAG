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
_LOCAL_CONFIG_PATH = _REPO_ROOT / "config" / "local.yaml"


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
class OcrConfig:
    enabled: bool
    lang: str
    use_gpu: bool
    min_text_chars: int
    max_pdf_pages: int
    skip_photo_folders: bool
    photo_folder_image_ratio: float
    photo_folder_min_files: int


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
    ocr: OcrConfig


@dataclass(frozen=True)
class ChunkingConfig:
    size: int
    overlap: int


@dataclass(frozen=True)
class RagConfig:
    top_k: int
    temperature: float
    hybrid_enabled: bool = True
    hybrid_fetch_multiplier: int = 4
    hybrid_keyword_weight: float = 0.45


@dataclass(frozen=True)
class ProfilesConfig:
    org_dir: str
    users_root: str
    user_id: str
    always_include_max_chars: int


@dataclass(frozen=True)
class WebConfig:
    enabled: bool
    score_threshold: float
    max_results: int
    fetch_top_n: int


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool
    propose_after_answer: bool


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
    profiles: ProfilesConfig
    web: WebConfig
    memory: MemoryConfig
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


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge top-level sections; nested dicts are updated key-wise."""
    result: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            merged = dict(result[key])
            merged.update(value)
            result[key] = merged
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return raw


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply INFRARAG_* environment overrides onto a loaded YAML dict."""
    data = {
        "app": dict(raw.get("app") or {}),
        "ollama": dict(raw.get("ollama") or {}),
        "chroma": dict(raw.get("chroma") or {}),
        "ingest": dict(raw.get("ingest") or {}),
        "chunking": dict(raw.get("chunking") or {}),
        "rag": dict(raw.get("rag") or {}),
        "profiles": dict(raw.get("profiles") or {}),
        "web": dict(raw.get("web") or {}),
        "memory": dict(raw.get("memory") or {}),
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
    if (v := _env("INFRARAG_USER_ID")) is not None:
        data["profiles"]["user_id"] = v

    return data


def _parse_ocr(raw: dict[str, Any] | None) -> OcrConfig:
    ocr = dict(raw or {})
    return OcrConfig(
        enabled=bool(ocr.get("enabled", True)),
        lang=str(ocr.get("lang", "es")),
        use_gpu=bool(ocr.get("use_gpu", False)),
        min_text_chars=int(ocr.get("min_text_chars", 40)),
        max_pdf_pages=int(ocr.get("max_pdf_pages", 50)),
        skip_photo_folders=bool(ocr.get("skip_photo_folders", True)),
        photo_folder_image_ratio=float(ocr.get("photo_folder_image_ratio", 0.6)),
        photo_folder_min_files=int(ocr.get("photo_folder_min_files", 5)),
    )


def _build_config(data: dict[str, Any], config_path: Path) -> AppConfig:
    app = data["app"]
    ollama = data["ollama"]
    chroma = data["chroma"]
    ingest = data["ingest"]
    chunking = data["chunking"]
    rag = data["rag"]
    profiles = data.get("profiles") or {}
    web = data.get("web") or {}
    memory = data.get("memory") or {}
    attachments = data["attachments"]
    ui = data["ui"]

    source_dir = ingest.get("source_dir")
    if source_dir is not None:
        source_dir = str(source_dir)

    ocr_raw = ingest.get("ocr")
    if ocr_raw is not None and not isinstance(ocr_raw, dict):
        raise ValueError("ingest.ocr must be a mapping")

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
            ocr=_parse_ocr(ocr_raw if isinstance(ocr_raw, dict) else None),
        ),
        chunking=ChunkingConfig(
            size=int(chunking["size"]),
            overlap=int(chunking["overlap"]),
        ),
        rag=RagConfig(
            top_k=int(rag["top_k"]),
            temperature=float(rag["temperature"]),
            hybrid_enabled=bool(rag.get("hybrid_enabled", True)),
            hybrid_fetch_multiplier=int(rag.get("hybrid_fetch_multiplier", 4)),
            hybrid_keyword_weight=float(rag.get("hybrid_keyword_weight", 0.35)),
        ),
        profiles=ProfilesConfig(
            org_dir=str(profiles.get("org_dir", "profiles/org")),
            users_root=str(profiles.get("users_root", "profiles/users")),
            user_id=str(profiles.get("user_id", "default")),
            always_include_max_chars=int(profiles.get("always_include_max_chars", 8000)),
        ),
        web=WebConfig(
            enabled=bool(web.get("enabled", True)),
            score_threshold=float(web.get("score_threshold", 0.5)),
            max_results=int(web.get("max_results", 5)),
            fetch_top_n=int(web.get("fetch_top_n", 2)),
        ),
        memory=MemoryConfig(
            enabled=bool(memory.get("enabled", True)),
            propose_after_answer=bool(memory.get("propose_after_answer", True)),
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
    """Load default.yaml, merge local.yaml, apply env overrides."""
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        raw = _load_yaml(config_path)
        data = _apply_env_overrides(raw)
        return _build_config(data, config_path.resolve())

    if not _DEFAULT_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config file not found: {_DEFAULT_CONFIG_PATH}")

    raw = _load_yaml(_DEFAULT_CONFIG_PATH)
    local = _load_yaml(_LOCAL_CONFIG_PATH)
    if local:
        raw = _deep_merge(raw, local)
    data = _apply_env_overrides(raw)
    return _build_config(data, _DEFAULT_CONFIG_PATH.resolve())


def write_local_ollama_config(
    *,
    chat_model: str,
    embed_model: str,
    path: Path | None = None,
) -> Path:
    """Write config/local.yaml with chosen Ollama models (used by bootstrap)."""
    target = path or _LOCAL_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ollama": {
            "chat_model": chat_model,
            "embed_model": embed_model,
        }
    }
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)
    return target
