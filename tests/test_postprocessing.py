"""Tests for app/postprocessing.py — Reciprocal Rank Fusion."""

from __future__ import annotations

from app.postprocessing import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        sem = [(0, 0.9), (1, 0.8), (2, 0.7)]
        kw = [(1, 5.0), (3, 4.0), (0, 3.0)]
        results = reciprocal_rank_fusion(sem, kw)
        # Chunk 1 appears in both lists — should rank high
        indices = [idx for idx, _ in results]
        assert 1 in indices
        assert 0 in indices

    def test_overlapping_chunks_boosted(self):
        sem = [(0, 0.9), (1, 0.7)]
        kw = [(0, 5.0), (2, 3.0)]
        results = reciprocal_rank_fusion(sem, kw, top_k=3)
        # Chunk 0 appears in both lists and ranks first in both — should be top
        assert results[0][0] == 0

    def test_non_overlapping(self):
        sem = [(0, 0.9)]
        kw = [(1, 5.0)]
        results = reciprocal_rank_fusion(sem, kw, top_k=5)
        indices = {idx for idx, _ in results}
        assert indices == {0, 1}

    def test_top_k_limits(self):
        sem = [(i, 1.0 - i * 0.1) for i in range(10)]
        kw = [(i + 5, 1.0 - i * 0.1) for i in range(10)]
        results = reciprocal_rank_fusion(sem, kw, top_k=3)
        assert len(results) == 3

    def test_empty_semantic(self):
        results = reciprocal_rank_fusion([], [(0, 1.0), (1, 0.5)], top_k=5)
        assert len(results) == 2

    def test_empty_keyword(self):
        results = reciprocal_rank_fusion([(0, 0.9)], [], top_k=5)
        assert len(results) == 1
        assert results[0][0] == 0

    def test_both_empty(self):
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_custom_k_parameter(self):
        sem = [(0, 0.9)]
        kw = [(0, 5.0)]
        # With k=1: score = 1/(1+1) + 1/(1+1) = 1.0
        results = reciprocal_rank_fusion(sem, kw, k=1, top_k=1)
        assert len(results) == 1
        assert results[0][1] == pytest.approx(1.0)


import pytest
