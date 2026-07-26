"""HTTP client for a local Ollama instance."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from infrarag.config import OllamaConfig


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API."""

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._base = config.base_url.rstrip("/")
        self._timeout = httpx.Timeout(config.timeout_s)

    def list_models(self) -> list[str]:
        """Return model names reported by the Ollama server."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self._base}/api/tags")
            response.raise_for_status()
            payload = response.json()
        return [m.get("name", "") for m in payload.get("models", []) if m.get("name")]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with the configured embedding model."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self._timeout) as client:
            for text in texts:
                response = client.post(
                    f"{self._base}/api/embeddings",
                    json={"model": self._config.embed_model, "prompt": text},
                )
                if response.status_code == 404:
                    # Newer Ollama API
                    response = client.post(
                        f"{self._base}/api/embed",
                        json={"model": self._config.embed_model, "input": text},
                    )
                response.raise_for_status()
                payload = response.json()
                if "embedding" in payload:
                    vectors.append(list(payload["embedding"]))
                elif "embeddings" in payload:
                    emb = payload["embeddings"]
                    vectors.append(list(emb[0] if emb and isinstance(emb[0], list) else emb))
                else:
                    raise RuntimeError(f"Unexpected embeddings response: {payload.keys()}")
        return vectors

    def chat(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        """Run a non-streaming chat completion."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base}/api/chat",
                json={
                    "model": self._config.chat_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            response.raise_for_status()
            payload = response.json()
        message = payload.get("message") or {}
        return str(message.get("content", ""))

    def chat_stream(
        self, messages: list[dict[str, str]], *, temperature: float
    ) -> Iterator[str]:
        """Yield chat completion tokens as they arrive."""
        with httpx.Client(timeout=self._timeout) as client:
            with client.stream(
                "POST",
                f"{self._base}/api/chat",
                json={
                    "model": self._config.chat_model,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": temperature},
                },
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    message = payload.get("message") or {}
                    content = message.get("content")
                    if content:
                        yield str(content)
                    if payload.get("done"):
                        break
