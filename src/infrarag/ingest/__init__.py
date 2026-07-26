"""Document ingest: walk directories, load files, chunk, differential index."""

from infrarag.models import IngestReport

__all__ = ["IngestReport", "ingest_directory"]


def __getattr__(name: str):
    if name == "ingest_directory":
        from infrarag.ingest.pipeline import ingest_directory

        return ingest_directory
    raise AttributeError(name)
