"""Vector store stubs."""

__all__ = ["ChromaStore"]


def __getattr__(name: str):
    if name == "ChromaStore":
        from infrarag.store.chroma_store import ChromaStore

        return ChromaStore
    raise AttributeError(name)
