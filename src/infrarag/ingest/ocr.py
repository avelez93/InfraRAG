"""Local PaddleOCR helpers (optional dependency: pip install -e '.[ocr]')."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from infrarag.config import OcrConfig

_ENGINE: Any | None = None
_ENGINE_KEY: tuple[str, bool] | None = None

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})


class OcrUnavailableError(RuntimeError):
    """Raised when OCR is required but the optional extra is not installed."""


def pdf_text_needs_ocr(text: str, *, page_count: int, min_text_chars: int) -> bool:
    """Return True when native PDF text is too thin to trust."""
    stripped = text.strip()
    if len(stripped) < min_text_chars:
        return True
    pages = max(page_count, 1)
    if (len(stripped) / pages) < 20:
        return True
    return False


def _require_ocr_packages() -> None:
    try:
        import paddleocr  # noqa: F401
        import pypdfium2  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        raise OcrUnavailableError(
            "OCR is required for this file but optional deps are missing. "
            "Install with: pip install -e '.[ocr]'"
        ) from exc


def _get_engine(config: OcrConfig) -> Any:
    global _ENGINE, _ENGINE_KEY
    key = (config.lang, config.use_gpu)
    if _ENGINE is not None and _ENGINE_KEY == key:
        return _ENGINE
    _require_ocr_packages()
    from paddleocr import PaddleOCR

    _ENGINE = PaddleOCR(
        use_angle_cls=True,
        lang=config.lang,
        use_gpu=config.use_gpu,
        show_log=False,
    )
    _ENGINE_KEY = key
    return _ENGINE


def _lines_from_paddle_result(result: Any) -> str:
    """Normalize PaddleOCR output to plain text."""
    lines: list[str] = []
    if not result:
        return ""
    for block in result:
        if not block:
            continue
        for item in block:
            # Typical: [box, (text, confidence)]
            if not item or len(item) < 2:
                continue
            payload = item[1]
            if isinstance(payload, (list, tuple)) and payload:
                lines.append(str(payload[0]))
            elif isinstance(payload, str):
                lines.append(payload)
    return "\n".join(lines).strip()


def ocr_image(path: Path, config: OcrConfig) -> str:
    """Run PaddleOCR on a single image file."""
    if not config.enabled:
        raise OcrUnavailableError("OCR is disabled in config (ingest.ocr.enabled=false)")
    engine = _get_engine(config)
    result = engine.ocr(str(path), cls=True)
    text = _lines_from_paddle_result(result)
    if not text:
        raise ValueError(f"OCR produced no text for {path}")
    return text


def ocr_pil_images(images: list[Any], config: OcrConfig) -> str:
    """Run PaddleOCR on in-memory PIL images (e.g. PDF pages)."""
    if not config.enabled:
        raise OcrUnavailableError("OCR is disabled in config (ingest.ocr.enabled=false)")
    if not images:
        return ""
    engine = _get_engine(config)
    import numpy as np

    parts: list[str] = []
    for image in images:
        arr = np.array(image.convert("RGB"))
        result = engine.ocr(arr, cls=True)
        part = _lines_from_paddle_result(result)
        if part:
            parts.append(part)
    return "\n\n".join(parts).strip()


def render_pdf_pages(path: Path, *, max_pages: int, scale: float = 2.0) -> list[Any]:
    """Render PDF pages to PIL images via pypdfium2."""
    _require_ocr_packages()
    import pypdfium2 as pdfium
    from PIL import Image

    pdf = pdfium.PdfDocument(str(path))
    images: list[Any] = []
    limit = min(len(pdf), max_pages)
    for i in range(limit):
        page = pdf[i]
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(pil)
        images.append(pil)
    return images
