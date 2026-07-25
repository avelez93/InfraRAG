"""ChromaDB-backed vector store (stub)."""

from __future__ import annotations

from infrarag.config import ChromaConfig
from infrarag.models import Chunk, QueryResult


class ChromaStore:
    """Persistent vector collection for corpus chunks.

    Implementation is deferred; methods raise NotImplementedError.
    """

    def __init__(self, config: ChromaConfig) -> None:
        self._config = config

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks and their embeddings into the collection."""
        raise NotImplementedError("ChromaStore.add_chunks is not implemented yet")

    def delete_by_source_path(self, source_path: str) -> None:
        """Remove all chunks belonging to a source file path."""
        raise NotImplementedError(
            "ChromaStore.delete_by_source_path is not implemented yet"
        )

    def query(
        self, embedding: list[float], *, top_k: int
    ) -> list[QueryResult]:
        """Return the top_k most similar chunks for an embedding."""
        raise NotImplementedError("ChromaStore.query is not implemented yet")
