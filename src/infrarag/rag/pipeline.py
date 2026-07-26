"""End-to-end ask / search pipeline."""

from __future__ import annotations

from collections.abc import Iterator

from infrarag.config import AppConfig
from infrarag.llm.ollama_client import OllamaClient
from infrarag.models import AttachmentContext, QueryResult
from infrarag.rag.retrieve import GatheredContext, gather_context, retrieve


def _build_messages(query: str, gathered: GatheredContext) -> list[dict[str, str]]:
    system = (
        "You are InfraRAG, a helpful assistant. "
        "The USER PROFILE describes the person asking; ORG PROFILE is shared organization context. "
        "Use profile, local corpus ([local]), web ([web]), and attachment context when relevant. "
        "Combine local facts (e.g. payroll documents) with web rates or rules when both appear. "
        "If context is insufficient, say so. Cite source paths and URLs when relevant. "
        "Always reply in the same language the user used in their question "
        "(for example Spanish if they wrote in Spanish, English if they wrote in English). "
        "For amounts, dates, account identifiers, and other figures: only state values that "
        "appear in the provided context. If a figure is missing, say so clearly. "
        "Do not invent, estimate, or average numbers across years or documents unless the "
        "context itself supports that calculation."
    )
    user = f"Context:\n{gathered.context_text}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def ask(
    query: str,
    config: AppConfig,
    attachments: AttachmentContext | None = None,
    *,
    mode: str = "ask",
) -> str:
    """Retrieve context (local + optional web + profiles), then generate an answer."""
    gathered = gather_context(query, config, mode=mode, attachments=attachments)
    messages = _build_messages(query, gathered)
    llm = OllamaClient(config.ollama)
    answer = llm.chat(messages, temperature=config.rag.temperature)
    if gathered.citations:
        cited = "\n".join(f"- {c}" for c in gathered.citations)
        return f"{answer.strip()}\n\nSources:\n{cited}"
    return answer.strip()


def ask_stream(
    query: str,
    config: AppConfig,
    attachments: AttachmentContext | None = None,
    *,
    mode: str = "ask",
) -> tuple[Iterator[str], GatheredContext]:
    """Stream answer tokens; return gathered context (citations, web flag) for the UI."""
    gathered = gather_context(query, config, mode=mode, attachments=attachments)
    messages = _build_messages(query, gathered)
    llm = OllamaClient(config.ollama)
    return llm.chat_stream(messages, temperature=config.rag.temperature), gathered


def search(query: str, config: AppConfig, *, top_k: int | None = None) -> list[QueryResult]:
    """Return ranked corpus hits without LLM generation."""
    return retrieve(query, config, top_k=top_k)
