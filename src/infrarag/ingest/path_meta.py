"""Extract year and query tokens for hybrid retrieval metadata."""

from __future__ import annotations

import re
from pathlib import Path

_YEAR_SEGMENT = re.compile(r"^(19|20)\d{2}$")
_YEAR_IN_TEXT = re.compile(r"(?<![0-9])((?:19|20)\d{2})(?![0-9])")
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def extract_year_from_path(path: Path | str) -> str | None:
    """Return the first path segment that is a 19xx/20xx year, else None."""
    p = Path(path)
    for part in p.parts:
        if _YEAR_SEGMENT.match(part):
            return part
    # Also check filename stem pieces like report_2024.pdf
    stem = p.stem
    match = _YEAR_IN_TEXT.search(stem)
    if match:
        return match.group(1)
    return None


def extract_years_from_query(query: str) -> list[str]:
    """Return distinct years mentioned in the query, in order of appearance."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _YEAR_IN_TEXT.finditer(query or ""):
        year = match.group(1)
        if year not in seen:
            seen.add(year)
            out.append(year)
    return out


def tokenize_query(query: str, *, min_len: int = 3) -> list[str]:
    """Lowercase alphanumeric tokens of at least min_len characters."""
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN.finditer(query or ""):
        tok = match.group(0).casefold()
        if len(tok) < min_len:
            continue
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    return tokens
