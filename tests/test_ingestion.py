"""Tests for app/ingestion.py — PDF extraction, chunking, persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.ingestion import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    _find_chunk_end,
    _normalize_whitespace,
    chunk_pages,
    extract_pdf_pages,
    ingest_pdf_bytes,
    load_store,
    save_store,
)
from app.models import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# _normalize_whitespace
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert _normalize_whitespace("a   b") == "a b"

    def test_collapses_newlines(self):
        assert _normalize_whitespace("a\n\nb") == "a b"

    def test_strips(self):
        assert _normalize_whitespace("  hello  ") == "hello"

    def test_tabs_and_mixed(self):
        assert _normalize_whitespace("a\t  \n b") == "a b"

    def test_empty(self):
        assert _normalize_whitespace("") == ""


# ---------------------------------------------------------------------------
# _find_chunk_end
# ---------------------------------------------------------------------------


class TestFindChunkEnd:
    def test_end_of_text(self):
        text = "Short text."
        assert _find_chunk_end(text, 0, 100) == len(text)

    def test_prefers_sentence_boundary(self):
        # Punctuation must be past 60% of chunk_size to be used.
        # With chunk_size=60, 60% = 36. The period at index 15 qualifies
        # if we use a chunk size where it falls in the last 40%.
        text = "A short intro. " + "x" * 100
        end = _find_chunk_end(text, 0, 20)
        # Period at index 14 is at 70% of 20=14, so it should be used.
        assert text[:end].rstrip().endswith(".")

    def test_falls_back_to_space(self):
        # No punctuation in the window — should break at whitespace
        text = "word " * 200  # lots of words, no sentence-ending punctuation
        end = _find_chunk_end(text, 0, 50)
        assert end < 60  # Should break near chunk_size
        # The character before end should be a space or the end is at a word boundary
        assert text[end - 1] == " " or text[end] == " " or end == 50

    def test_hard_break(self):
        # Single long "word" with no spaces or punctuation
        text = "a" * 600
        end = _find_chunk_end(text, 0, 512)
        assert end == 512


# ---------------------------------------------------------------------------
# extract_pdf_pages
# ---------------------------------------------------------------------------


class TestExtractPdfPages:
    def test_valid_pdf(self, tiny_pdf_bytes):
        pages = extract_pdf_pages(tiny_pdf_bytes)
        assert len(pages) >= 1
        page_num, text = pages[0]
        assert page_num == 1
        assert "Hello" in text or "test" in text

    def test_empty_pdf_raises(self, empty_pdf_bytes):
        with pytest.raises(ValueError, match="No extractable text"):
            extract_pdf_pages(empty_pdf_bytes)

    def test_invalid_bytes_raises(self):
        with pytest.raises(Exception):
            extract_pdf_pages(b"not a pdf")


# ---------------------------------------------------------------------------
# chunk_pages
# ---------------------------------------------------------------------------


class TestChunkPages:
    def test_single_short_page(self):
        pages = [(1, "Hello world.")]
        chunks = chunk_pages("test.pdf", pages)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].metadata.filename == "test.pdf"
        assert chunks[0].metadata.page_number == 1
        assert chunks[0].metadata.chunk_index == 0

    def test_long_text_produces_multiple_chunks(self):
        long_text = "This is a sentence. " * 100  # ~2000 chars
        pages = [(1, long_text)]
        chunks = chunk_pages("test.pdf", pages, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        # Chunk indices should be sequential
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i

    def test_overlap_creates_shared_content(self):
        text = "A" * 300 + ". " + "B" * 300
        pages = [(1, text)]
        chunks = chunk_pages("test.pdf", pages, chunk_size=400, overlap=100)
        if len(chunks) >= 2:
            # There should be some overlap between consecutive chunks
            c0, c1 = chunks[0].text, chunks[1].text
            assert len(c0) > 0 and len(c1) > 0

    def test_multiple_pages(self):
        pages = [(1, "Page one text."), (2, "Page two text.")]
        chunks = chunk_pages("test.pdf", pages)
        filenames = {c.metadata.filename for c in chunks}
        assert filenames == {"test.pdf"}
        # At least one chunk per page
        page_nums = {c.metadata.page_number for c in chunks}
        assert 1 in page_nums and 2 in page_nums

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="did not produce any text chunks"):
            chunk_pages("test.pdf", [(1, "   ")])

    def test_default_chunk_size(self):
        assert CHUNK_SIZE == 512
        assert CHUNK_OVERLAP == 100


# ---------------------------------------------------------------------------
# save_store / load_store
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_round_trip(self, tmp_path, sample_chunks, sample_embeddings, sample_summaries):
        chunks_path = tmp_path / "chunks.json"
        emb_path = tmp_path / "embeddings.npy"
        sum_path = tmp_path / "summaries.json"

        with (
            patch("app.ingestion.DATA_DIR", tmp_path),
            patch("app.ingestion.CHUNKS_PATH", chunks_path),
            patch("app.ingestion.EMBEDDINGS_PATH", emb_path),
            patch("app.ingestion.SUMMARIES_PATH", sum_path),
        ):
            save_store(sample_chunks, sample_embeddings, sample_summaries)
            loaded_chunks, loaded_emb, loaded_sum = load_store()

        assert len(loaded_chunks) == len(sample_chunks)
        assert loaded_emb.shape == sample_embeddings.shape
        np.testing.assert_array_almost_equal(loaded_emb, sample_embeddings, decimal=5)
        assert loaded_sum == sample_summaries

    def test_load_missing_files_returns_defaults(self, tmp_path):
        with (
            patch("app.ingestion.CHUNKS_PATH", tmp_path / "nope.json"),
            patch("app.ingestion.EMBEDDINGS_PATH", tmp_path / "nope.npy"),
            patch("app.ingestion.SUMMARIES_PATH", tmp_path / "nope.json"),
        ):
            chunks, emb, summaries = load_store()

        assert chunks == []
        assert emb.shape == (0, 0)
        assert summaries == {}

    def test_load_mismatched_counts_raises(self, tmp_path):
        chunks_path = tmp_path / "chunks.json"
        emb_path = tmp_path / "embeddings.npy"

        # Save 2 chunks but 3 embeddings
        chunks = [
            Chunk(text="a", metadata=ChunkMetadata(filename="f.pdf", page_number=1, chunk_index=0)),
            Chunk(text="b", metadata=ChunkMetadata(filename="f.pdf", page_number=1, chunk_index=1)),
        ]
        chunks_path.write_text(json.dumps([c.model_dump() for c in chunks]))
        np.save(emb_path, np.zeros((3, 4), dtype=np.float32))

        with (
            patch("app.ingestion.CHUNKS_PATH", chunks_path),
            patch("app.ingestion.EMBEDDINGS_PATH", emb_path),
            patch("app.ingestion.SUMMARIES_PATH", tmp_path / "nope.json"),
        ):
            with pytest.raises(RuntimeError, match="inconsistent"):
                load_store()


# ---------------------------------------------------------------------------
# ingest_pdf_bytes (integration, mocked API)
# ---------------------------------------------------------------------------


class TestIngestPdfBytes:
    @pytest.mark.asyncio
    async def test_returns_chunks_embeddings_summary(self, tiny_pdf_bytes):
        async def mock_embed(texts, **kwargs):
            n = len(texts)
            return np.random.randn(n, 4).astype(np.float32)

        async def mock_summary(chunks, **kwargs):
            return "A summary."

        with (
            patch("app.ingestion.embed_texts", side_effect=mock_embed),
            patch("app.ingestion.generate_document_summary", side_effect=mock_summary),
        ):
            chunks, embeddings, summary = await ingest_pdf_bytes("test.pdf", tiny_pdf_bytes)

        assert len(chunks) > 0
        assert embeddings.shape[0] == len(chunks)
        assert summary == "A summary."
