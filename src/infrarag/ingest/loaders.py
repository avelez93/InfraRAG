"""Load text from supported document formats (stub)."""

from __future__ import annotations

from pathlib import Path

from infrarag.models import Document

# Families planned for implementation:
# - text: .txt, .md
# - pdf: .pdf (pypdf)
# - office: .docx, .xlsx, .pptx
# - libreoffice: .odt, .ods, .odp (odfpy)


def load_document(path: Path) -> Document:
    """Extract plain text from a supported file path."""
    raise NotImplementedError(f"load_document is not implemented yet: {path}")
