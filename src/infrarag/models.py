"""Shared domain models for InfraRAG."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A loaded source document before chunking."""

    source_path: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A text chunk ready for embedding / storage."""

    chunk_id: str
    source_path: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResult:
    """A retrieved chunk with similarity score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class IngestReport:
    """Summary of a differential directory ingest run."""

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AttachmentContext:
    """Ephemeral turn context from chat file/URL attachments."""

    sources: list[str]
    text: str


@dataclass(frozen=True)
class WebHit:
    """A single web search result, optionally with fetched page text."""

    title: str
    url: str
    snippet: str
    page_text: str = ""


@dataclass(frozen=True)
class ProposedFact:
    """A durable fact proposed for confirmed user memory."""

    summary: str
    text: str
