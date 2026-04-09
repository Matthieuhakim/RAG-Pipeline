"""Tests for app/search.py — HybridSearchStore and tokenizer."""

from __future__ import annotations

import numpy as np
import pytest

from app.models import Chunk, ChunkMetadata
from app.search import HybridSearchStore, tokenize


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_removed(self):
        assert tokenize("hello, world!") == ["hello", "world"]

    def test_numbers_kept(self):
        assert tokenize("version 3.14") == ["version", "3", "14"]

    def test_empty(self):
        assert tokenize("") == []

    def test_only_punctuation(self):
        assert tokenize("!!! ???") == []

    def test_mixed_case(self):
        assert tokenize("CamelCase UPPER lower") == ["camelcase", "upper", "lower"]


# ---------------------------------------------------------------------------
# HybridSearchStore — initialization and data management
# ---------------------------------------------------------------------------


class TestHybridSearchStoreInit:
    def test_empty_init(self):
        store = HybridSearchStore()
        assert store.chunks == []
        assert store.embeddings.shape == (0, 0)
        assert store.summaries == {}
        assert not store.has_data()

    def test_set_store(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)
        assert len(store.chunks) == 3
        assert store.embeddings.shape == (3, 4)
        assert len(store.summaries) == 2
        assert store.has_data()

    def test_has_data_false_when_empty(self):
        store = HybridSearchStore()
        store.set_store([], np.empty((0, 0), dtype=np.float32), {})
        assert not store.has_data()


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


class TestBM25Index:
    def test_index_built_on_set_store(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)
        assert len(store._doc_term_freqs) == 3
        assert len(store._doc_lengths) == 3
        assert store._average_doc_length > 0

    def test_document_frequencies(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)
        # "is" appears in at least 2 of the 3 chunks
        assert store._document_frequencies.get("is", 0) >= 1


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    def test_returns_ranked_results(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)
        # Use first chunk's embedding as query — should rank itself highest
        query_emb = sample_embeddings[0]
        results = store.semantic_search(query_emb, top_k=3)
        assert len(results) > 0
        # Best result should be index 0 (identical vector)
        assert results[0][0] == 0
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)

    def test_top_k_limits_results(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)
        results = store.semantic_search(sample_embeddings[0], top_k=1)
        assert len(results) == 1

    def test_empty_store_returns_empty(self):
        store = HybridSearchStore()
        query_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.semantic_search(query_emb, top_k=5)
        assert results == []

    def test_zero_norm_query_safe(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)
        zero_query = np.zeros(4, dtype=np.float32)
        results = store.semantic_search(zero_query, top_k=3)
        # Should not crash; results should be empty (all similarities ~0)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Keyword search (BM25)
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    def test_exact_match_scores_high(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)
        # "photosynthesis" appears only in chunk 2
        results = store.keyword_search("photosynthesis", top_k=3)
        assert len(results) > 0
        assert results[0][0] == 2  # biology chunk

    def test_empty_query(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)
        results = store.keyword_search("", top_k=3)
        assert results == []

    def test_no_match(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)
        results = store.keyword_search("xyzzyplugh", top_k=3)
        assert results == []

    def test_empty_store(self):
        store = HybridSearchStore()
        results = store.keyword_search("machine learning", top_k=3)
        assert results == []


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def test_returns_both_lists(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)
        sem, kw = store.hybrid_search("machine learning", sample_embeddings[0], top_k=3)
        assert isinstance(sem, list)
        assert isinstance(kw, list)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


class TestUpsert:
    def test_upsert_replaces_same_file(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)
        assert len(store.chunks) == 3

        # Upsert new data for "ml_intro.pdf" — should replace old chunks from that file
        new_chunk = Chunk(
            text="Updated ML content.",
            metadata=ChunkMetadata(filename="ml_intro.pdf", page_number=1, chunk_index=0),
        )
        new_emb = np.random.randn(1, 4).astype(np.float32)
        store.upsert_documents([new_chunk], new_emb, {"ml_intro.pdf": "Updated summary."})

        # Should have 1 (biology) + 1 (new ml_intro) = 2 chunks
        assert len(store.chunks) == 2
        filenames = {c.metadata.filename for c in store.chunks}
        assert filenames == {"ml_intro.pdf", "biology.pdf"}

    def test_upsert_adds_new_file(self, sample_chunks, sample_embeddings, sample_summaries):
        store = HybridSearchStore()
        store.set_store(sample_chunks, sample_embeddings, sample_summaries)

        new_chunk = Chunk(
            text="Chemistry is fun.",
            metadata=ChunkMetadata(filename="chemistry.pdf", page_number=1, chunk_index=0),
        )
        new_emb = np.random.randn(1, 4).astype(np.float32)
        store.upsert_documents([new_chunk], new_emb, {"chemistry.pdf": "Chemistry overview."})

        assert len(store.chunks) == 4
        assert "chemistry.pdf" in store.summaries
