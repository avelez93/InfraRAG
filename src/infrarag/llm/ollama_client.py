"""HTTP client for a local Ollama instance."""

from __future__ import annotations

from collections.abc import Iterator

from infrarag.config import OllamaConfig


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API.

    Implementation is deferred to a later phase; methods raise NotImplementedError.
    """

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config

    def list_models(self) -> list[str]:
        """Return model names reported by the Ollama server."""
        raise NotImplementedError("OllamaClient.list_models is not implemented yet")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with the configured embedding model."""
        raise NotImplementedError("OllamaClient.embed is not implemented yet")

    def chat(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        """Run a non-streaming chat completion."""
        raise NotImplementedError("OllamaClient.chat is not implemented yet")

    def chat_stream(
        self, messages: list[dict[str, str]], *, temperature: float
    ) -> Iterator[str]:
        """Yield chat completion tokens as they arrive."""
        raise NotImplementedError("OllamaClient.chat_stream is not implemented yet")
