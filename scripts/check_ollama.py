#!/usr/bin/env python3
"""Check that a local Ollama server is reachable."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

# Allow running without install: add src/ to path when needed
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from infrarag.config import load_config  # noqa: E402


def main() -> int:
    config = load_config()
    url = f"{config.ollama.base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(url, timeout=min(10, config.ollama.timeout_s))
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Ollama check failed: {exc}", file=sys.stderr)
        print(f"Tried GET {url}", file=sys.stderr)
        print("Is `ollama serve` running?", file=sys.stderr)
        return 1

    payload = response.json()
    models = [m.get("name", "") for m in payload.get("models", [])]
    print(f"Ollama OK at {config.ollama.base_url}")
    if models:
        print("Models:")
        for name in models:
            print(f"  - {name}")
    else:
        print("No models pulled yet. Suggested:")
        print(f"  ollama pull {config.ollama.chat_model}")
        print(f"  ollama pull {config.ollama.embed_model}")

    chat = config.ollama.chat_model
    embed = config.ollama.embed_model
    # Ollama may report "name:tag" or with digest suffixes; match by prefix
    def _present(wanted: str) -> bool:
        return any(m == wanted or m.startswith(f"{wanted}") for m in models)

    missing = [m for m in (chat, embed) if not _present(m)]
    if missing:
        print("Configured models not found locally:")
        for name in missing:
            print(f"  ollama pull {name}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
