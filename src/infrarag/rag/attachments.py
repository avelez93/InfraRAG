"""Resolve ephemeral chat attachments: local files and web URLs (stub)."""

from __future__ import annotations

from pathlib import Path

from infrarag.config import AppConfig
from infrarag.models import AttachmentContext


def resolve_attachments(
    files: list[Path],
    urls: list[str],
    config: AppConfig,
) -> AttachmentContext:
    """Load file text and fetch URL text into a single turn context.

    Attachments are ephemeral by default and are not written to the corpus.
    """
    raise NotImplementedError("resolve_attachments is not implemented yet")
