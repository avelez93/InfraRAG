"""Directory ingest orchestration."""

from __future__ import annotations

from pathlib import Path

from infrarag.config import AppConfig
from infrarag.ingest.chunking import chunk_document
from infrarag.ingest.differential import (
    classify_change,
    fingerprint_file,
    load_manifest,
    save_manifest,
)
from infrarag.ingest.loaders import load_document
from infrarag.ingest.path_meta import extract_year_from_path
from infrarag.ingest.roots import (
    infer_roots_from_source_paths,
    load_ingest_roots,
    record_ingest_root,
    save_ingest_roots,
)
from infrarag.ingest.walker import iter_source_files
from infrarag.llm.ollama_client import OllamaClient
from infrarag.models import Chunk, IngestReport
from infrarag.store.chroma_store import ChromaStore


def ingest_directory(
    path: Path,
    config: AppConfig,
    *,
    force: bool = False,
) -> IngestReport:
    """Walk path, load readable files, differentially chunk and index into Chroma.

    When ``force`` is True, every readable file is re-chunked and re-embedded
    even if the differential manifest marks it unchanged.
    """
    root = path.expanduser().resolve()
    store = ChromaStore(config.chroma)
    llm = OllamaClient(config.ollama)
    manifest_path = Path(config.ingest.manifest_path)
    if not manifest_path.is_absolute():
        # Resolve relative to repo-ish cwd; prefer absolute paths in production.
        manifest_path = Path.cwd() / manifest_path

    previous = load_manifest(manifest_path) if config.ingest.differential else {}
    next_manifest = dict(previous)

    added = updated = unchanged = skipped = 0
    errors: list[str] = []

    try:
        files = list(iter_source_files(root, config.ingest))
    except OSError as exc:
        return IngestReport(errors=[str(exc)])

    for file_path in files:
        try:
            fp = fingerprint_file(file_path, root=root)
            if force:
                change = "updated" if previous.get(fp.relative_path) is not None else "added"
            elif config.ingest.differential:
                change = classify_change(fp, previous.get(fp.relative_path))
            else:
                change = "added"
            if change == "unchanged":
                unchanged += 1
                continue

            document = load_document(file_path, ocr=config.ingest.ocr)
            chunks = chunk_document(document, config.chunking)
            if not chunks:
                skipped += 1
                continue

            embeddings = llm.embed([c.text for c in chunks])
            source_key = str(file_path.resolve())
            year = extract_year_from_path(file_path)
            chunks = [
                Chunk(
                    chunk_id=c.chunk_id,
                    source_path=source_key,
                    text=c.text,
                    metadata={
                        **c.metadata,
                        **({"year": year} if year else {}),
                    },
                )
                for c in chunks
            ]
            store.delete_by_source_path(source_key)
            store.add_chunks(chunks, embeddings)
            next_manifest[fp.relative_path] = fp
            if change == "added":
                added += 1
            else:
                updated += 1
        except Exception as exc:  # noqa: BLE001 - collect per-file errors
            errors.append(f"{file_path}: {exc}")
            skipped += 1

    if config.ingest.differential or next_manifest:
        save_manifest(manifest_path, next_manifest)

    if root.is_dir():
        record_ingest_root(config, root)

    return IngestReport(
        added=added,
        updated=updated,
        unchanged=unchanged,
        skipped=skipped,
        errors=errors,
    )


def resolve_known_ingest_roots(config: AppConfig) -> list[Path]:
    """Known ingest directories: saved roots, else inferred from Chroma source paths."""
    roots = [r for r in load_ingest_roots(config) if r.is_dir()]
    if roots:
        return roots
    store = ChromaStore(config.chroma)
    inferred = infer_roots_from_source_paths(store.list_source_paths())
    existing = [r for r in inferred if r.is_dir()]
    if existing:
        save_ingest_roots(config, existing)
    return existing


def reingest_known_roots(config: AppConfig) -> list[tuple[Path, IngestReport]]:
    """Force re-index every known ingest root directory."""
    roots = resolve_known_ingest_roots(config)
    results: list[tuple[Path, IngestReport]] = []
    for root in roots:
        report = ingest_directory(root, config, force=True)
        results.append((root, report))
    return results
