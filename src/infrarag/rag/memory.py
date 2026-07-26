"""Propose and persist confirmed user-memory facts."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from infrarag.config import AppConfig
from infrarag.ingest.pipeline import ingest_directory
from infrarag.llm.ollama_client import OllamaClient
from infrarag.models import IngestReport, ProposedFact
from infrarag.rag.profiles import (
    load_known_memory_text,
    load_profile_context,
    resolve_memory_dir,
    resolve_user_dir,
)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def normalize_fact_text(text: str) -> str:
    """Collapse whitespace and casefold for duplicate detection."""
    return " ".join(text.casefold().split())


def _token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.casefold()) if len(tok) > 2}


def is_duplicate_fact(fact: ProposedFact, known_blob: str) -> bool:
    """Return True if fact summary/text is already covered by known profile/memory text."""
    known = normalize_fact_text(known_blob)
    if not known:
        return False
    for candidate in (fact.text, fact.summary):
        norm = normalize_fact_text(candidate)
        if not norm:
            continue
        if len(norm) >= 12 and norm in known:
            return True
        tokens = _token_set(norm)
        if len(tokens) >= 4:
            known_tokens = _token_set(known)
            overlap = len(tokens & known_tokens) / len(tokens)
            if overlap >= 0.85:
                return True
    return False


def dedupe_facts(facts: list[ProposedFact], known_blob: str) -> list[ProposedFact]:
    """Drop facts already present in known_blob and near-duplicates within the list."""
    kept: list[ProposedFact] = []
    running = known_blob
    for fact in facts:
        summary = fact.summary.strip()
        text = fact.text.strip()
        if not summary or not text:
            continue
        candidate = ProposedFact(summary=summary, text=text)
        if is_duplicate_fact(candidate, running):
            continue
        kept.append(candidate)
        running = f"{running}\n{candidate.summary}\n{candidate.text}"
    return kept


def build_known_blob(config: AppConfig, *, base: Path | None = None) -> str:
    """Combine always-include profiles and existing memory files."""
    profile = load_profile_context(config, base=base)
    memory = load_known_memory_text(config, base=base)
    parts = [profile.org_text, profile.user_text, memory]
    return "\n\n".join(p for p in parts if p and p.strip())


def _extract_json_object(raw: str) -> dict | None:
    text = raw.strip()
    fence = _JSON_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def propose_facts(
    *,
    query: str,
    answer: str,
    config: AppConfig,
    base: Path | None = None,
) -> list[ProposedFact]:
    """Ask the LLM for durable facts; return only non-duplicate proposals."""
    if not config.memory.enabled or not config.memory.propose_after_answer:
        return []

    known = build_known_blob(config, base=base)
    system = (
        "You extract durable personal or organization facts worth saving for future answers. "
        "Reply with JSON only, no markdown prose outside JSON. "
        'Schema: {"facts": [{"summary": "...", "text": "..."}], "reason": "optional"}. '
        "Include identity, relationships, job/role, preferences, standing conventions, "
        "stable constraints. "
        "Do NOT include one-off calculations, generic trivia, secrets/passwords/tokens, "
        "or anything already stated in ALREADY_KNOWN. "
        "If nothing new and durable, return {\"facts\": []}."
    )
    user = (
        f"ALREADY_KNOWN:\n{known or '(empty)'}\n\n"
        f"USER_QUESTION:\n{query}\n\n"
        f"ASSISTANT_ANSWER:\n{answer}\n"
    )
    llm = OllamaClient(config.ollama)
    raw = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
    )
    payload = _extract_json_object(raw)
    if not payload:
        return []
    raw_facts = payload.get("facts")
    if not isinstance(raw_facts, list):
        return []
    parsed: list[ProposedFact] = []
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or "").strip()
        text = str(item.get("text") or "").strip()
        if summary and text:
            parsed.append(ProposedFact(summary=summary, text=text))
    return dedupe_facts(parsed, known)


def format_memory_markdown(facts: list[ProposedFact], *, when: datetime | None = None) -> str:
    """Render confirmed facts as a markdown memory note."""
    stamp = when or datetime.now(UTC)
    lines = [
        f"# Confirmed memory - {stamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]
    for fact in facts:
        lines.append(f"## {fact.summary}")
        lines.append("")
        lines.append(fact.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_confirmed_facts(
    facts: list[ProposedFact],
    config: AppConfig,
    *,
    base: Path | None = None,
    ingest: bool = True,
) -> tuple[Path | None, IngestReport | None]:
    """Dedupe again, write a memory markdown file, optionally differential-ingest."""
    known = build_known_blob(config, base=base)
    remaining = dedupe_facts(facts, known)
    if not remaining:
        return None, None

    memory_dir = resolve_memory_dir(config, base=base)
    memory_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC)
    filename = stamp.strftime("%Y%m%d-%H%M%S.md")
    path = memory_dir / filename
    path.write_text(format_memory_markdown(remaining, when=stamp), encoding="utf-8")

    report: IngestReport | None = None
    if ingest:
        user_dir = resolve_user_dir(config, base=base)
        report = ingest_directory(user_dir, config)
    return path, report
