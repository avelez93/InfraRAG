"""ChromaDB-backed vector store."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from infrarag.config import ChromaConfig
from infrarag.models import Chunk, QueryResult


class ChromaStore:
    """Persistent vector collection for corpus chunks."""

    def __init__(self, config: ChromaConfig) -> None:
        self._config = config
        persist = Path(config.persist_dir)
        persist.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=config.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks and their embeddings into the collection."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_path": c.source_path,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in chunks
            ],
        )

    def delete_by_source_path(self, source_path: str) -> None:
        """Remove all chunks belonging to a source file path."""
        self._collection.delete(where={"source_path": source_path})

    def list_source_paths(self) -> list[str]:
        """Return distinct absolute source_path values stored in the collection."""
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception:  # noqa: BLE001
            return []
        metas = result.get("metadatas") or []
        seen: set[str] = set()
        out: list[str] = []
        for meta in metas:
            if not meta:
                continue
            path = str(meta.get("source_path") or "").strip()
            if path and path not in seen:
                seen.add(path)
                out.append(path)
        out.sort()
        return out

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int,
        where: dict[str, str] | None = None,
    ) -> list[QueryResult]:
        """Return the top_k most similar chunks for an embedding."""
        if top_k <= 0:
            return []
        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        try:
            result = self._collection.query(**kwargs)
        except Exception:
            if where:
                # Old indexes or missing metadata fields: fall back unfiltered.
                result = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
            else:
                raise
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        out: list[QueryResult] = []
        for i, chunk_id in enumerate(ids):
            meta = dict(metas[i] or {})
            source_path = str(meta.pop("source_path", ""))
            text = docs[i] or ""
            distance = float(dists[i]) if i < len(dists) else 0.0
            # Cosine distance -> similarity-ish score
            score = 1.0 - distance
            out.append(
                QueryResult(
                    chunk=Chunk(
                        chunk_id=str(chunk_id),
                        source_path=source_path,
                        text=text,
                        metadata={str(k): str(v) for k, v in meta.items()},
                    ),
                    score=score,
                )
            )
        return out
