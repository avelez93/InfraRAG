"""Hardware-based chat model tier definitions (single source of truth)."""

from __future__ import annotations

from dataclasses import dataclass


EMBED_MODEL = "nomic-embed-text"


@dataclass(frozen=True)
class ModelTier:
    """One selectable chat-model size category."""

    id: str
    chat_model: str
    params_label: str
    download_gb: float
    memory_gb_label: str
    suggested_machine: str
    notes: str
    min_effective_gb: float  # inclusive lower bound for auto-select


# Ordered ascending by capacity. Thresholds: select highest tier where effective_gb >= min.
TIERS: tuple[ModelTier, ...] = (
    ModelTier(
        id="nano",
        chat_model="qwen3:0.6b",
        params_label="0.6B",
        download_gb=0.5,
        memory_gb_label="1-2 GB",
        suggested_machine="4-8 GB RAM, or any small GPU",
        notes="Fastest, weakest",
        min_effective_gb=0.0,
    ),
    ModelTier(
        id="tiny",
        chat_model="qwen3:1.7b",
        params_label="1.7B",
        download_gb=1.2,
        memory_gb_label="2-3 GB",
        suggested_machine="8 GB RAM, or 4 GB VRAM",
        notes="Light RAG; GPU OK",
        min_effective_gb=4.0,
    ),
    ModelTier(
        id="compact",
        chat_model="qwen2.5:3b",
        params_label="3B",
        download_gb=2.0,
        memory_gb_label="3-4 GB",
        suggested_machine="8-16 GB RAM, or 4-6 GB VRAM",
        notes="Intermediate tiny/small; scaffolding default",
        min_effective_gb=6.0,
    ),
    ModelTier(
        id="small",
        chat_model="qwen3:4b",
        params_label="4B",
        download_gb=2.5,
        memory_gb_label="3-5 GB",
        suggested_machine="16 GB RAM, or 6 GB+ VRAM",
        notes="Newer Qwen3; GPU OK",
        min_effective_gb=9.0,
    ),
    ModelTier(
        id="medium",
        chat_model="qwen3:8b",
        params_label="8B",
        download_gb=5.0,
        memory_gb_label="6-8 GB",
        suggested_machine="16-32 GB RAM, or 8-12 GB VRAM",
        notes="Strong multilingual/docs",
        min_effective_gb=14.0,
    ),
    ModelTier(
        id="large",
        chat_model="qwen3:14b",
        params_label="14B",
        download_gb=9.0,
        memory_gb_label="10-14 GB",
        suggested_machine="32 GB+ RAM, or 16 GB VRAM",
        notes="High quality auto pick",
        min_effective_gb=24.0,
    ),
    ModelTier(
        id="xlarge",
        chat_model="qwen3:32b",
        params_label="32B",
        download_gb=20.0,
        memory_gb_label="20-24 GB",
        suggested_machine="64 GB RAM, or 24 GB+ VRAM",
        notes="Best local quality in the set",
        min_effective_gb=40.0,
    ),
)


def get_tier(tier_id: str) -> ModelTier:
    """Return a tier by id (case-insensitive)."""
    key = tier_id.strip().lower()
    for tier in TIERS:
        if tier.id == key:
            return tier
    raise KeyError(f"Unknown tier id: {tier_id}")


def select_tier(effective_gb: float) -> ModelTier:
    """Pick the highest tier whose min_effective_gb <= effective_gb."""
    chosen = TIERS[0]
    for tier in TIERS:
        if effective_gb >= tier.min_effective_gb:
            chosen = tier
    return chosen


def format_tiers_table(*, recommended_id: str | None = None) -> str:
    """ASCII table of all tiers for bootstrap / README-style display."""
    headers = (
        f"{'':1} {'Tier':<10} {'Model':<14} {'Memory':<12} {'Download':<10} {'Notes'}"
    )
    lines = [headers, "-" * len(headers)]
    for tier in TIERS:
        mark = "*" if recommended_id and tier.id == recommended_id else " "
        note = tier.notes
        if recommended_id and tier.id == recommended_id:
            note = f"{note} <- recommended"
        lines.append(
            f"{mark} {tier.id:<10} {tier.chat_model:<14} "
            f"{tier.memory_gb_label:<12} ~{tier.download_gb} GB{'':<4} {note}"
        )
    return "\n".join(lines)
