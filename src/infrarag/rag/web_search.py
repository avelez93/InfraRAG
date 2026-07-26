"""DuckDuckGo web search and optional page fetch for complementary context."""

from __future__ import annotations

import logging

from infrarag.config import AppConfig
from infrarag.models import WebHit
from infrarag.rag.attachments import fetch_url_text

logger = logging.getLogger(__name__)


def search_web(query: str, config: AppConfig) -> list[WebHit]:
    """Search the web and optionally fetch top page bodies."""
    query = query.strip()
    if not query or not config.web.enabled:
        return []

    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError(
            "duckduckgo-search is required for web search; install project dependencies"
        ) from exc

    max_results = max(0, config.web.max_results)
    if max_results == 0:
        return []

    raw_hits: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("href") or item.get("link") or "").strip()
                if not url:
                    continue
                raw_hits.append(
                    {
                        "title": str(item.get("title") or "").strip(),
                        "url": url,
                        "snippet": str(item.get("body") or item.get("snippet") or "").strip(),
                    }
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Web search failed: %s", exc)
        return []

    fetch_n = min(max(0, config.web.fetch_top_n), len(raw_hits))
    timeout_s = config.attachments.url_timeout_s
    hits: list[WebHit] = []
    for i, item in enumerate(raw_hits):
        page_text = ""
        if i < fetch_n:
            try:
                page_text = fetch_url_text(item["url"], timeout_s=timeout_s)
            except Exception as exc:  # noqa: BLE001
                logger.info("Failed to fetch web page %s: %s", item["url"], exc)
        hits.append(
            WebHit(
                title=item["title"],
                url=item["url"],
                snippet=item["snippet"],
                page_text=page_text,
            )
        )
    return hits


def format_web_context(hits: list[WebHit]) -> str:
    """Render web hits as a labeled context block."""
    if not hits:
        return ""
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        body = hit.page_text.strip() or hit.snippet.strip()
        title = hit.title or hit.url
        blocks.append(f"[{i}] title={title}\nurl={hit.url}\n{body}")
    return "[web]\n" + "\n\n".join(blocks)
