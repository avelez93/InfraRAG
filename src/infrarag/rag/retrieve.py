"""Semantic retrieval and context gathering for Ask / Ask Deep."""

from __future__ import annotations

from dataclasses import dataclass, replace

from infrarag.config import AppConfig
from infrarag.ingest.path_meta import (
    extract_years_from_query,
    tokenize_query,
)
from infrarag.llm.ollama_client import OllamaClient
from infrarag.models import AttachmentContext, QueryResult, WebHit
from infrarag.rag.profiles import ProfileContext, load_profile_context
from infrarag.rag.web_search import format_web_context, search_web
from infrarag.store.chroma_store import ChromaStore

_MAX_FETCH = 40
_YEAR_MATCH_BONUS = 0.5


@dataclass(frozen=True)
class GatheredContext:
    """Merged local, web, profile, and attachment context for Ask."""

    local_hits: list[QueryResult]
    web_hits: list[WebHit]
    profile: ProfileContext
    attachments: AttachmentContext | None
    max_local_score: float
    web_used: bool
    context_text: str
    citations: list[str]
    local_citations: list[str]
    web_citations: list[str]


def keyword_score(
    hit: QueryResult,
    *,
    tokens: list[str],
    query_years: list[str],
) -> float:
    """Overlap of query tokens against path+text, with year match bonus."""
    blob = f"{hit.chunk.source_path}\n{hit.chunk.text}".casefold()
    meta_year = str(hit.chunk.metadata.get("year") or "")
    score = 0.0
    if tokens:
        hits = sum(1 for tok in tokens if tok in blob)
        score = hits / len(tokens)
    if query_years:
        year_ok = False
        for year in query_years:
            if year == meta_year or year in blob:
                year_ok = True
                break
        if year_ok:
            score = min(1.0, score + _YEAR_MATCH_BONUS)
    return score


def hit_matches_year(hit: QueryResult, query_years: list[str]) -> bool:
    if not query_years:
        return True
    meta_year = str(hit.chunk.metadata.get("year") or "")
    blob = f"{hit.chunk.source_path}\n{hit.chunk.text}"
    return any(y == meta_year or y in blob for y in query_years)


def hybrid_rescore(
    hits: list[QueryResult],
    *,
    query: str,
    top_k: int,
    keyword_weight: float,
) -> list[QueryResult]:
    """Combine vector scores with keyword/year overlap and optionally prefer year hits."""
    if not hits or top_k <= 0:
        return []
    tokens = tokenize_query(query)
    query_years = extract_years_from_query(query)
    kw_w = max(0.0, min(1.0, keyword_weight))
    vec_w = 1.0 - kw_w

    rescored: list[QueryResult] = []
    for hit in hits:
        kw = keyword_score(hit, tokens=tokens, query_years=query_years)
        combined = vec_w * hit.score + kw_w * kw
        rescored.append(replace(hit, score=combined))

    rescored.sort(key=lambda h: h.score, reverse=True)

    if query_years:
        matching = [h for h in rescored if hit_matches_year(h, query_years)]
        min_keep = min(top_k, 3)
        if len(matching) >= min_keep:
            return matching[:top_k]
    return rescored[:top_k]


def _merge_hits(*groups: list[QueryResult]) -> list[QueryResult]:
    by_id: dict[str, QueryResult] = {}
    for group in groups:
        for hit in group:
            existing = by_id.get(hit.chunk.chunk_id)
            if existing is None or hit.score > existing.score:
                by_id[hit.chunk.chunk_id] = hit
    return list(by_id.values())


def retrieve(query: str, config: AppConfig, *, top_k: int | None = None) -> list[QueryResult]:
    """Embed the query and return the top_k most similar corpus chunks (hybrid optional)."""
    k = config.rag.top_k if top_k is None else top_k
    llm = OllamaClient(config.ollama)
    store = ChromaStore(config.chroma)
    vectors = llm.embed([query])
    if not vectors:
        return []
    embedding = vectors[0]

    if not config.rag.hybrid_enabled:
        return store.query(embedding, top_k=k)

    fetch_n = min(_MAX_FETCH, max(k, k * max(1, config.rag.hybrid_fetch_multiplier)))
    years = extract_years_from_query(query)

    unfiltered = store.query(embedding, top_k=fetch_n)
    filtered: list[QueryResult] = []
    if len(years) == 1:
        filtered = store.query(embedding, top_k=fetch_n, where={"year": years[0]})

    merged = _merge_hits(filtered, unfiltered)
    return hybrid_rescore(
        merged,
        query=query,
        top_k=k,
        keyword_weight=config.rag.hybrid_keyword_weight,
    )


def max_score(hits: list[QueryResult]) -> float:
    """Return the highest similarity score among hits (0.0 if empty)."""
    if not hits:
        return 0.0
    return max(hit.score for hit in hits)


def should_use_web(*, mode: str, max_local_score: float, config: AppConfig) -> bool:
    """Decide whether to complement with web search."""
    if not config.web.enabled:
        return False
    if mode == "deep":
        return True
    if mode == "ask":
        return max_local_score <= config.web.score_threshold
    return False


def gather_context(
    query: str,
    config: AppConfig,
    *,
    mode: str = "ask",
    attachments: AttachmentContext | None = None,
) -> GatheredContext:
    """Load profiles, retrieve locally, and optionally complement with web."""
    profile = load_profile_context(config)
    local_hits = retrieve(query, config)
    local_max = max_score(local_hits)
    use_web = should_use_web(mode=mode, max_local_score=local_max, config=config)
    web_hits: list[WebHit] = []
    if use_web:
        try:
            web_hits = search_web(query, config)
        except Exception:  # noqa: BLE001
            web_hits = []

    blocks: list[str] = []
    local_citations: list[str] = []
    web_citations: list[str] = []

    if profile.combined_text.strip():
        blocks.append(profile.combined_text.strip())

    if local_hits:
        local_parts: list[str] = []
        for i, hit in enumerate(local_hits, start=1):
            local_citations.append(hit.chunk.source_path)
            local_parts.append(f"[{i}] source={hit.chunk.source_path}\n{hit.chunk.text}")
        blocks.append("[local]\n" + "\n\n".join(local_parts))

    web_block = format_web_context(web_hits)
    if web_block:
        blocks.append(web_block)
        web_citations.extend(hit.url for hit in web_hits)

    if attachments and attachments.text.strip():
        blocks.append(f"[attachments]\n{attachments.text}")
        local_citations.extend(attachments.sources)

    context_text = "\n\n".join(blocks) if blocks else "(no context retrieved)"

    seen: set[str] = set()
    citations: list[str] = []
    for path in [*profile.org_sources, *profile.user_sources, *local_citations, *web_citations]:
        if path and path not in seen:
            seen.add(path)
            citations.append(path)

    return GatheredContext(
        local_hits=local_hits,
        web_hits=web_hits,
        profile=profile,
        attachments=attachments,
        max_local_score=local_max,
        web_used=bool(web_hits) or use_web,
        context_text=context_text,
        citations=citations,
        local_citations=_unique(local_citations),
        web_citations=_unique(web_citations),
    )


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
