"""Tests for bonus/similarity_threshold.py."""

from __future__ import annotations

from bonus.similarity_threshold import (
    DEFAULT_SIMILARITY_THRESHOLD,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    check_similarity_threshold,
)


class TestCheckSimilarityThreshold:
    def test_empty_results_returns_false(self):
        assert check_similarity_threshold([]) is False

    def test_below_threshold(self):
        results = [(0, 0.5), (1, 0.3)]
        assert check_similarity_threshold(results) is False

    def test_at_threshold(self):
        results = [(0, DEFAULT_SIMILARITY_THRESHOLD)]
        assert check_similarity_threshold(results) is True

    def test_above_threshold(self):
        results = [(0, 0.95), (1, 0.80)]
        assert check_similarity_threshold(results) is True

    def test_custom_threshold(self):
        results = [(0, 0.5)]
        assert check_similarity_threshold(results, threshold=0.4) is True
        assert check_similarity_threshold(results, threshold=0.6) is False

    def test_uses_max_score(self):
        results = [(0, 0.3), (1, 0.9), (2, 0.2)]
        assert check_similarity_threshold(results) is True


class TestConstants:
    def test_default_threshold(self):
        assert DEFAULT_SIMILARITY_THRESHOLD == 0.60

    def test_insufficient_evidence_message(self):
        assert "sufficient evidence" in INSUFFICIENT_EVIDENCE_MESSAGE.lower()
