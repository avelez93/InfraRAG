"""Document ingest: walk directories, load files, chunk, differential index."""

from infrarag.ingest.pipeline import ingest_directory
from infrarag.models import IngestReport

__all__ = ["IngestReport", "ingest_directory"]
