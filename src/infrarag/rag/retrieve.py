"""Semantic retrieval over the vector store (stub)."""

from __future__ import annotations

from infrarag.config import AppConfig
from infrarag.models import QueryResult


def retrieve(query: str, config: AppConfig, *, top_k: int | None = None) -> list[QueryResult]:
    """Embed the query and return the top_k most similar corpus chunks."""
    raise NotImplementedError("retrieve is not implemented yet")
