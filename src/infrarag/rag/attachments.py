"""Resolve ephemeral chat attachments: local files and web URLs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from infrarag.config import AppConfig
from infrarag.ingest.loaders import load_document
from infrarag.models import AttachmentContext


def resolve_attachments(
    files: list[Path],
    urls: list[str],
    config: AppConfig,
) -> AttachmentContext:
    """Load file text and fetch URL text into a single turn context."""
    sources: list[str] = []
    parts: list[str] = []

    for path in files:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Attachment not found: {path}")
        size = path.stat().st_size
        if size > config.attachments.max_file_bytes:
            raise ValueError(f"Attachment too large: {path}")
        doc = load_document(path, ocr=config.ingest.ocr)
        sources.append(str(path))
        parts.append(f"[Attachment: {path.name}]\n{doc.text}")

    for url in urls:
        url = url.strip()
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in config.attachments.allowed_url_schemes:
            raise ValueError(f"URL scheme not allowed: {url}")
        text = fetch_url_text(url, timeout_s=config.attachments.url_timeout_s)
        sources.append(url)
        parts.append(f"[URL: {url}]\n{text}")

    return AttachmentContext(sources=sources, text="\n\n".join(parts))


def fetch_url_text(url: str, *, timeout_s: int) -> str:
    """Fetch a URL and return plain text (HTML stripped when applicable)."""
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        raw = response.text
    if (
        "html" in content_type
        or raw.lstrip().lower().startswith("<!doctype")
        or "<html" in raw.lower()
    ):
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = raw
    # Bound context size
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())[:50000]
