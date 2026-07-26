"""Ollama client."""

__all__ = ["OllamaClient"]


def __getattr__(name: str):
    if name == "OllamaClient":
        from infrarag.llm.ollama_client import OllamaClient

        return OllamaClient
    raise AttributeError(name)
