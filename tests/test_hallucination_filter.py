"""Tests for bonus/hallucination_filter.py."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app.models import Chunk, ChunkMetadata
from bonus.hallucination_filter import (
    HALLUCINATION_WARNING,
    _cosine_similarities,
    _is_structural,
    _split_sentences,
    filter_hallucinations,
)


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_basic(self):
        result = _split_sentences("Hello world. Goodbye world.")
        assert result == ["Hello world.", "Goodbye world."]

    def test_multiple_delimiters(self):
        result = _split_sentences("Yes! No? Maybe.")
        assert len(result) == 3

    def test_single_sentence(self):
        result = _split_sentences("Just one sentence.")
        assert result == ["Just one sentence."]

    def test_empty(self):
        result = _split_sentences("")
        assert result == []

    def test_whitespace_only(self):
        result = _split_sentences("   ")
        assert result == []

    def test_preserves_content(self):
        text = "First. Second. Third."
        result = _split_sentences(text)
        assert len(result) == 3
        assert result[0] == "First."


# ---------------------------------------------------------------------------
# _is_structural
# ---------------------------------------------------------------------------


class TestIsStructural:
    def test_short_sentence(self):
        assert _is_structural("OK") is True
        assert _is_structural("Yes") is True

    def test_based_on_context(self):
        assert _is_structural("Based on the provided context, we see that...") is True

    def test_here_is(self):
        assert _is_structural("Here is the answer to your question.") is True

    def test_in_summary(self):
        assert _is_structural("In summary, the results show improvement.") is True

    def test_factual_claim(self):
        assert _is_structural("The efficiency of monocrystalline panels is 20%.") is False

    def test_according_to(self):
        assert _is_structural("According to the provided context, results vary.") is True


# ---------------------------------------------------------------------------
# _cosine_similarities
# ---------------------------------------------------------------------------


class TestCosineSimilarities:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        chunks = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sims = _cosine_similarities(v, chunks)
        assert sims[0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        chunks = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        sims = _cosine_similarities(v, chunks)
        assert sims[0] == pytest.approx(0.0, abs=1e-6)

    def test_multiple_chunks(self):
        v = np.array([1.0, 0.0], dtype=np.float32)
        chunks = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        sims = _cosine_similarities(v, chunks)
        assert len(sims) == 3
        assert sims[0] > sims[1]

    def test_zero_vector_safe(self):
        v = np.zeros(3, dtype=np.float32)
        chunks = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sims = _cosine_similarities(v, chunks)
        # Should not crash; result should be ~0
        assert abs(sims[0]) < 1e-6


# ---------------------------------------------------------------------------
# filter_hallucinations
# ---------------------------------------------------------------------------


def _make_chunks_and_embeddings():
    """Create chunks about ML with corresponding embeddings."""
    chunks = [
        Chunk(text="Machine learning uses data.", metadata=ChunkMetadata(filename="f.pdf", page_number=1, chunk_index=0)),
        Chunk(text="Neural networks learn patterns.", metadata=ChunkMetadata(filename="f.pdf", page_number=1, chunk_index=1)),
    ]
    # 6D vectors with orthogonal directions: dims 0-1 for chunk0, dims 2-3 for chunk1.
    # Unsupported content will embed along dims 4-5 (orthogonal to both).
    emb = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    return chunks, emb


class TestFilterHallucinations:
    @pytest.mark.asyncio
    async def test_all_supported_no_change(self):
        """When all sentences are supported, answer is unchanged."""
        chunks, emb = _make_chunks_and_embeddings()

        # Mock embed_texts to return embeddings very similar to chunk 0
        async def mock_embed(texts, **kwargs):
            # Return embeddings aligned with chunk 0 for each text
            n = len(texts)
            vecs = np.tile(emb[0], (n, 1))
            return vecs

        with patch("bonus.hallucination_filter.embed_texts", side_effect=mock_embed):
            answer = "Machine learning uses data to learn."
            result, was_filtered = await filter_hallucinations(answer, chunks, emb)

        assert not was_filtered
        assert result == answer

    @pytest.mark.asyncio
    async def test_unsupported_stripped(self):
        """Unsupported sentences should be removed with a warning."""
        chunks, emb = _make_chunks_and_embeddings()

        call_count = [0]

        async def mock_embed(texts, **kwargs):
            # Return embeddings: first text aligned with chunks, second orthogonal
            results = []
            for t in texts:
                if "machine learning" in t.lower() or "neural" in t.lower():
                    results.append(emb[0])
                else:
                    # Orthogonal to both chunks (along dim 4-5)
                    results.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32))
            return np.array(results, dtype=np.float32)

        with patch("bonus.hallucination_filter.embed_texts", side_effect=mock_embed):
            answer = "Machine learning uses data for training. The price of Bitcoin reached a hundred thousand dollars last year. Neural networks learn complex patterns from examples."
            result, was_filtered = await filter_hallucinations(answer, chunks, emb, threshold=0.5)

        assert was_filtered
        assert "Bitcoin" not in result
        assert HALLUCINATION_WARNING in result

    @pytest.mark.asyncio
    async def test_empty_chunks_unchanged(self):
        """With no chunks, the answer should be returned unchanged."""
        answer = "Some text here."
        result, was_filtered = await filter_hallucinations(
            answer, [], np.empty((0, 6), dtype=np.float32)
        )
        assert not was_filtered
        assert result == answer

    @pytest.mark.asyncio
    async def test_structural_only_unchanged(self):
        """An answer with only structural sentences is not filtered."""
        chunks, emb = _make_chunks_and_embeddings()
        answer = "Based on the context, here is the result."
        result, was_filtered = await filter_hallucinations(answer, chunks, emb)
        assert not was_filtered

    @pytest.mark.asyncio
    async def test_all_unsupported_fallback(self):
        """When all claims are unsupported, return fallback message."""
        chunks, emb = _make_chunks_and_embeddings()

        async def mock_embed(texts, **kwargs):
            n = len(texts)
            # All orthogonal to chunks (along dim 4-5)
            return np.tile(
                np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), (n, 1)
            )

        with patch("bonus.hallucination_filter.embed_texts", side_effect=mock_embed):
            answer = "Aliens reportedly landed on the surface of Mars last Tuesday. Bitcoin has become a widely adopted digital currency worldwide."
            result, was_filtered = await filter_hallucinations(answer, chunks, emb, threshold=0.5)

        assert was_filtered
        assert "unable to produce" in result.lower()
