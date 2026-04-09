"""Tests for app/models.py — Pydantic model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import (
    Chunk,
    ChunkMetadata,
    ErrorResponse,
    IngestResponse,
    IntentDecision,
    QueryRequest,
    QueryResponse,
    SourceReference,
    StatusResponse,
)


# ---------------------------------------------------------------------------
# ChunkMetadata
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    def test_valid(self):
        m = ChunkMetadata(filename="a.pdf", page_number=1, chunk_index=0)
        assert m.filename == "a.pdf"
        assert m.page_number == 1
        assert m.chunk_index == 0

    def test_page_number_zero_rejected(self):
        with pytest.raises(ValidationError):
            ChunkMetadata(filename="a.pdf", page_number=0, chunk_index=0)

    def test_page_number_negative_rejected(self):
        with pytest.raises(ValidationError):
            ChunkMetadata(filename="a.pdf", page_number=-1, chunk_index=0)

    def test_chunk_index_negative_rejected(self):
        with pytest.raises(ValidationError):
            ChunkMetadata(filename="a.pdf", page_number=1, chunk_index=-1)


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_valid(self):
        c = Chunk(
            text="hello",
            metadata=ChunkMetadata(filename="a.pdf", page_number=1, chunk_index=0),
        )
        assert c.text == "hello"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            Chunk(
                text="",
                metadata=ChunkMetadata(filename="a.pdf", page_number=1, chunk_index=0),
            )


# ---------------------------------------------------------------------------
# SourceReference
# ---------------------------------------------------------------------------


class TestSourceReference:
    def test_valid(self):
        s = SourceReference(filename="a.pdf", page_number=2, chunk_index=5)
        assert s.page_number == 2

    def test_page_number_zero_rejected(self):
        with pytest.raises(ValidationError):
            SourceReference(filename="a.pdf", page_number=0, chunk_index=0)


# ---------------------------------------------------------------------------
# QueryRequest
# ---------------------------------------------------------------------------


class TestQueryRequest:
    def test_valid(self):
        q = QueryRequest(query="hello")
        assert q.query == "hello"

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")


# ---------------------------------------------------------------------------
# QueryResponse
# ---------------------------------------------------------------------------


class TestQueryResponse:
    def test_defaults(self):
        r = QueryResponse(answer="yes", retrieval_used=False)
        assert r.sources == []

    def test_with_sources(self):
        s = SourceReference(filename="a.pdf", page_number=1, chunk_index=0)
        r = QueryResponse(answer="yes", sources=[s], retrieval_used=True)
        assert len(r.sources) == 1


# ---------------------------------------------------------------------------
# IngestResponse
# ---------------------------------------------------------------------------


class TestIngestResponse:
    def test_defaults(self):
        r = IngestResponse(status="success", chunk_count=10)
        assert r.filenames_processed == []
        assert r.errors == []

    def test_negative_chunk_count_rejected(self):
        with pytest.raises(ValidationError):
            IngestResponse(status="success", chunk_count=-1)


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------


class TestErrorResponse:
    def test_with_details(self):
        e = ErrorResponse(error_type="test", message="msg", details={"key": "val"})
        assert e.details == {"key": "val"}

    def test_details_none(self):
        e = ErrorResponse(error_type="test", message="msg")
        assert e.details is None


# ---------------------------------------------------------------------------
# StatusResponse
# ---------------------------------------------------------------------------


class TestStatusResponse:
    def test_defaults(self):
        s = StatusResponse(ingested_files=0, total_chunks=0)
        assert s.filenames == []

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            StatusResponse(ingested_files=-1, total_chunks=0)


# ---------------------------------------------------------------------------
# IntentDecision
# ---------------------------------------------------------------------------


class TestIntentDecision:
    def test_defaults(self):
        d = IntentDecision(needs_retrieval=True)
        assert d.reasoning == ""

    def test_with_reasoning(self):
        d = IntentDecision(needs_retrieval=False, reasoning="No docs")
        assert d.reasoning == "No docs"
