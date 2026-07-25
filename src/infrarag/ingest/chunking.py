"""Split documents into overlapping character chunks (stub)."""

from __future__ import annotations

from infrarag.config import ChunkingConfig
from infrarag.models import Chunk, Document


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Split document text using character size and overlap from config."""
    raise NotImplementedError("chunk_document is not implemented yet")
