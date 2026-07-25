# InfraRAG

Local, cross-platform RAG (Retrieval-Augmented Generation) in Python. Point InfraRAG at a directory of documents, differentially index what changed, then ask questions or search -- backed by [Ollama](https://ollama.com) and embedded [ChromaDB](https://www.trychroma.com/).

## Features (scaffolding)

This repository is currently a **scaffold**: package layout, configuration, UI shell, and stubs. The ingest/RAG pipeline is not wired yet -- see [FUTURE_STEPS.md](FUTURE_STEPS.md).

Planned behavior:

- **Differential directory ingest** -- recursively index readable files; on re-run, only new or modified files are re-indexed
- **Formats** -- `.txt`, `.md`, `.pdf`, Office (`.docx`, `.xlsx`, `.pptx`), LibreOffice (`.odt`, `.ods`, `.odp`)
- **Ask** -- chat with optional ephemeral file/URL attachments per turn
- **Search** -- semantic search over the corpus
- **Local-first** -- Ollama for chat + embeddings; Chroma persisted under `data/chroma`

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running (Linux, Windows, or macOS)
- Suggested models:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

## Install

```bash
cd InfraRAG
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` if you want environment overrides.

## Configure

Defaults live in [`config/default.yaml`](config/default.yaml) (single source of truth). Optional env overrides:

| Variable | Purpose |
|----------|---------|
| `INFRARAG_SOURCE_DIR` | Default directory to index |
| `INFRARAG_OLLAMA_BASE_URL` | Ollama HTTP base URL |
| `INFRARAG_CHAT_MODEL` | Chat model name |
| `INFRARAG_EMBED_MODEL` | Embedding model name |
| `INFRARAG_CHROMA_DIR` | Chroma persist directory |

## Run

```bash
# Verify Ollama is reachable
python scripts/check_ollama.py

# Launch the Streamlit UI (Index / Ask / Search stubs)
infrarag
# or
python -m infrarag
```

## Repository map

```
config/default.yaml     Runtime defaults
src/infrarag/           Application package
  config.py             YAML + env ? AppConfig
  llm/                  Ollama client (stub)
  store/                Chroma store (stub)
  ingest/               Walker, loaders, differential ingest (stubs)
  rag/                  Retrieve, ask, attachments (stubs)
  ui/app.py             Streamlit UI shell
data/chroma/            Vector DB + ingest manifest (local)
tests/                  Pytest
scripts/check_ollama.py Ollama connectivity check
FUTURE_STEPS.md         Roadmap beyond scaffolding
```

## Development

```bash
pytest
ruff check src tests
```

## License

Apache License 2.0 -- see [LICENSE](LICENSE).
