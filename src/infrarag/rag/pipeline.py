"""End-to-end ask / search pipeline (stub)."""

from __future__ import annotations

from infrarag.config import AppConfig
from infrarag.models import AttachmentContext, QueryResult


def ask(
    query: str,
    config: AppConfig,
    attachments: AttachmentContext | None = None,
) -> str:
    """Retrieve corpus context, merge optional attachments, and generate an answer."""
    raise NotImplementedError("ask is not implemented yet")


def search(query: str, config: AppConfig, *, top_k: int | None = None) -> list[QueryResult]:
    """Return ranked corpus hits without LLM generation."""
    raise NotImplementedError("search is not implemented yet")
