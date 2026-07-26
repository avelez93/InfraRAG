"""RAG retrieve / ask / attachment helpers."""

from infrarag.models import AttachmentContext, ProposedFact, QueryResult, WebHit
from infrarag.rag.attachments import resolve_attachments
from infrarag.rag.memory import propose_facts, save_confirmed_facts
from infrarag.rag.pipeline import ask, ask_stream, search
from infrarag.rag.retrieve import gather_context, retrieve

__all__ = [
    "AttachmentContext",
    "ProposedFact",
    "QueryResult",
    "WebHit",
    "ask",
    "ask_stream",
    "gather_context",
    "propose_facts",
    "resolve_attachments",
    "retrieve",
    "save_confirmed_facts",
    "search",
]
