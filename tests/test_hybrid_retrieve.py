"""Tests for year metadata helpers and hybrid rescoring."""

from __future__ import annotations

from pathlib import Path

from infrarag.ingest.path_meta import (
    extract_year_from_path,
    extract_years_from_query,
    tokenize_query,
)
from infrarag.models import Chunk, QueryResult
from infrarag.rag.retrieve import hit_matches_year, hybrid_rescore, keyword_score


def test_extract_year_from_path_segment() -> None:
    path = Path("/home/u/informes/2024/nomina.pdf")
    assert extract_year_from_path(path) == "2024"


def test_extract_year_from_filename() -> None:
    assert extract_year_from_path(Path("/docs/report_2023_final.pdf")) == "2023"


def test_extract_year_none() -> None:
    assert extract_year_from_path(Path("/docs/readme.md")) is None


def test_extract_years_from_query() -> None:
    assert extract_years_from_query("nómina 2024 y IRPF") == ["2024"]
    assert extract_years_from_query("compare 2023 vs 2024") == ["2023", "2024"]
    assert extract_years_from_query("hello") == []


def test_tokenize_query() -> None:
    tokens = tokenize_query("payroll IRPF 2024 x")
    assert "payroll" in tokens
    assert "2024" in tokens
    assert "irpf" in tokens
    assert "x" not in tokens  # too short


def test_keyword_score_year_bonus() -> None:
    hit = QueryResult(
        chunk=Chunk(
            chunk_id="1",
            source_path="/informes/2024/a.pdf",
            text="payroll",
            metadata={"year": "2024"},
        ),
        score=0.5,
    )
    score = keyword_score(hit, tokens=["payroll"], query_years=["2024"])
    assert score >= 0.5
    assert hit_matches_year(hit, ["2024"]) is True
    assert hit_matches_year(hit, ["2023"]) is False


def test_hybrid_rescore_prefers_matching_year() -> None:
    hits = [
        QueryResult(
            chunk=Chunk(
                chunk_id="old",
                source_path="/informes/2023/a.pdf",
                text="payslip amount 1000",
                metadata={"year": "2023"},
            ),
            score=0.95,
        ),
        QueryResult(
            chunk=Chunk(
                chunk_id="new",
                source_path="/informes/2024/b.pdf",
                text="payslip amount 1100",
                metadata={"year": "2024"},
            ),
            score=0.4,
        ),
        QueryResult(
            chunk=Chunk(
                chunk_id="new2",
                source_path="/informes/2024/c.pdf",
                text="payslip bonus",
                metadata={"year": "2024"},
            ),
            score=0.35,
        ),
        QueryResult(
            chunk=Chunk(
                chunk_id="new3",
                source_path="/informes/2024/d.pdf",
                text="payslip tax",
                metadata={"year": "2024"},
            ),
            score=0.3,
        ),
    ]
    ranked = hybrid_rescore(
        hits,
        query="payslip 2024",
        top_k=3,
        keyword_weight=0.35,
    )
    assert len(ranked) == 3
    assert all(hit_matches_year(h, ["2024"]) for h in ranked)
    assert all("2024" in h.chunk.source_path for h in ranked)


def test_hybrid_rescore_keeps_mixed_when_few_year_hits() -> None:
    hits = [
        QueryResult(
            chunk=Chunk(
                chunk_id="a",
                source_path="/other/doc.pdf",
                text="general note",
                metadata={},
            ),
            score=0.9,
        ),
        QueryResult(
            chunk=Chunk(
                chunk_id="b",
                source_path="/informes/2024/one.pdf",
                text="payslip",
                metadata={"year": "2024"},
            ),
            score=0.2,
        ),
    ]
    ranked = hybrid_rescore(
        hits,
        query="payslip 2024",
        top_k=2,
        keyword_weight=0.35,
    )
    # Only one year match (< min_keep=2), so mixed results remain.
    assert len(ranked) == 2
    ids = {h.chunk.chunk_id for h in ranked}
    assert ids == {"a", "b"}
