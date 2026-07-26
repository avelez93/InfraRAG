"""Split documents into overlapping character chunks."""

from __future__ import annotations

import hashlib

from infrarag.config import ChunkingConfig
from infrarag.models import Chunk, Document


def chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    """Split document text using character size and overlap from config."""
    size = config.size
    overlap = config.overlap
    if size <= 0:
        raise ValueError("chunking.size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunking.overlap must be >= 0 and < size")

    text = document.text
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = size - overlap
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end]
        digest = hashlib.sha1(
            f"{document.source_path}:{start}:{end}".encode()
        ).hexdigest()[:16]
        chunk_id = f"{digest}-{index}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_path=document.source_path,
                text=piece,
                metadata={
                    "start": str(start),
                    "end": str(end),
                    "index": str(index),
                },
            )
        )
        index += 1
        if end >= len(text):
            break
        start += step
    return chunks
