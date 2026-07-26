# InfraRAG

Local, cross-platform RAG (Retrieval-Augmented Generation) in Python. Point InfraRAG at a directory of documents, differentially index what changed, then ask questions or search -- backed by [Ollama](https://ollama.com) and embedded [ChromaDB](https://www.trychroma.com/).

## Quick start

```bash
# Linux / macOS
./scripts/install.sh

# Windows (PowerShell)
.\scripts\install.ps1

# Non-interactive (use hardware-recommended model)
./scripts/install.sh -y
```

The installer:

1. Creates `.venv` and installs the package
2. Detects RAM / GPU and recommends a chat-model tier
3. Shows the requirements table and lets you choose (Enter / `default` = recommended)
4. Writes `config/local.yaml`, tries to ensure Ollama, and pulls embed + chat models

Then:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
infrarag
```

## Model tiers

All tiers use the GPU automatically when Ollama supports it (CUDA / ROCm / Metal). Memory figures are approximate (Q4). Embedding model for all tiers: `nomic-embed-text`.

| Tier | Ollama tag | Params | Memory (RAM or VRAM) | Suggested machine |
|------|------------|--------|----------------------|-------------------|
| nano | qwen3:0.6b | 0.6B | 1-2 GB | 4-8 GB RAM |
| tiny | qwen3:1.7b | 1.7B | 2-3 GB | 8 GB RAM / 4 GB VRAM |
| compact | qwen2.5:3b | 3B | 3-4 GB | 8-16 GB RAM / 4-6 GB VRAM |
| small | qwen3:4b | 4B | 3-5 GB | 16 GB RAM / 6 GB+ VRAM |
| medium | qwen3:8b | 8B | 6-8 GB | 16-32 GB RAM / 8-12 GB VRAM |
| large | qwen3:14b | 14B | 10-14 GB | 32 GB+ RAM / 16 GB VRAM |
| xlarge | qwen3:32b | 32B | 20-24 GB | 64 GB RAM / 24 GB+ VRAM |

Override anytime via `INFRARAG_CHAT_MODEL` or edit `config/local.yaml`.

## Features

- **Differential directory ingest** -- recursively index readable files; re-runs only new/changed files
- **Formats** -- `.txt`, `.md`, `.pdf`, Office, LibreOffice, plus `.png`/`.jpg`/`.jpeg` via OCR
- **Ask** -- streaming chat with optional ephemeral file/URL attachments and source citations; web search complements when local max score ≤ 0.5
- **Ask Deep** -- always searches local corpus and the web (slower)
- **Profiles** -- org + user profile folders always injected into Ask; confirmed facts saved under user `memory/` after you approve
- **Search** -- semantic search over the corpus, with hybrid keyword + year-from-path boost
- **Local-first** -- Ollama for chat + embeddings; Chroma under `data/chroma`; DuckDuckGo for web (no API key)

## Hybrid retrieval and years

Ask/Search fetch a wider vector candidate set, then rescore with keyword overlap and **year** signals (path segment like `informes/2024/...`, filename, or chunk metadata). Mentions of a year in the question prefer matching folders when enough hits exist.

After upgrading, **re-index** your document directories so existing chunks get `year` metadata. Use **Index → Re-index all** to force re-embed every previously indexed directory (tracked in `data/chroma/ingest_roots.json`). Untouched differential files stay unchanged until you re-index or modify them.

## OCR (scanned PDFs and images)

Install the optional extra (large download; first run also fetches Paddle models):

```bash
pip install -e ".[ocr]"
# or with dev tools:
pip install -e ".[dev,ocr]"
```

InfraRAG pins **PaddleOCR 2.x** (`paddleocr>=2.7.3,<3`). PaddleOCR 3.x pulls `paddlex` / `pandas<=1.5.3`, which fails to build on **Python 3.12**. Prefer Python 3.11–3.12 with the pin above; if `paddlepaddle` itself has no wheel for your platform, use 3.11.

Verify:

```bash
python verifications/verify_ocr_install.py
```

Behavior (`ingest.ocr` in `config/default.yaml`):

- **PDF:** extract text with pypdf; if too thin/empty, render pages and run **PaddleOCR** locally
- **PNG/JPG/JPEG:** OCR as documents
- **Photo folders:** if a directory has at least 5 files and images are >= 60% of them, **skip image files** there (still index PDF/Office/txt in that folder)

Disable or tune via `ingest.ocr` (`enabled`, `lang`, `use_gpu`, `skip_photo_folders`, ratios, `max_pdf_pages`).

## Profiles and memory

Layout (templates are committed; your confirmed memory files are gitignored):

```
profiles/
  org/profile.md              # shared org context (always included in Ask)
  users/<user_id>/profile.md  # personal context (always included)
  users/<user_id>/memory/     # facts you confirm after an answer
```

Set `profiles.user_id` in config or `INFRARAG_USER_ID` (default `default`). Edit the markdown templates, then optionally **Index → Index org/user profile** so they are also searchable in Chroma. After Ask / Ask Deep, InfraRAG may propose durable facts; nothing is written until you confirm. Duplicates already in profile or memory are filtered out.

## Web complement

- **Ask:** retrieves the local corpus always; if `max(local score) ≤ web.score_threshold` (default `0.5`), also searches DuckDuckGo and fetches a few top pages.
- **Ask Deep:** always local + web (slower).
- Tunables: `web.enabled`, `web.score_threshold`, `web.max_results`, `web.fetch_top_n` in `config/default.yaml`.

## Configure

Defaults: [`config/default.yaml`](config/default.yaml). Machine overrides: `config/local.yaml` (gitignored, written by bootstrap). Env (highest priority):

| Variable | Purpose |
|----------|---------|
| `INFRARAG_SOURCE_DIR` | Default directory to index |
| `INFRARAG_OLLAMA_BASE_URL` | Ollama HTTP base URL |
| `INFRARAG_CHAT_MODEL` | Chat model name |
| `INFRARAG_EMBED_MODEL` | Embedding model name |
| `INFRARAG_CHROMA_DIR` | Chroma persist directory |
| `INFRARAG_USER_ID` | Active profile under `profiles/users/` |

## Manual install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/bootstrap.py --skip-install   # tier choice + Ollama pulls
python scripts/check_ollama.py
infrarag
```

## Repository map

```
config/default.yaml     Runtime defaults
config/local.yaml       Bootstrap-chosen models (local)
profiles/               Org + user profile templates and memory
src/infrarag/           Application package
scripts/install.sh      Linux/macOS installer
scripts/install.ps1     Windows installer
scripts/bootstrap.py    Bootstrap logic
scripts/check_ollama.py Ollama + model check
FUTURE_STEPS.md         Roadmap
```

## Development

```bash
pytest
ruff check src tests
```

## License

Apache License 2.0 -- see [LICENSE](LICENSE).

See also [FUTURE_STEPS.md](FUTURE_STEPS.md).
