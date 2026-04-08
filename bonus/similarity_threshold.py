"""Similarity threshold gate for retrieval results.

Refuses to answer when the top-k chunks do not meet a minimum cosine
similarity threshold, returning an "insufficient evidence" response
instead of risking a hallucinated answer.
"""

from __future__ import annotations

DEFAULT_SIMILARITY_THRESHOLD = 0.75

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have sufficient evidence in the ingested documents to answer "
    "this question confidently. The retrieved chunks did not meet the "
    "minimum relevance threshold. Please try rephrasing your question or "
    "uploading more relevant documents."
)


def check_similarity_threshold(
    semantic_results: list[tuple[int, float]],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> bool:
    """Return True if at least one semantic result meets the threshold.

    Parameters
    ----------
    semantic_results:
        List of (chunk_index, cosine_similarity) pairs from semantic search.
    threshold:
        Minimum cosine similarity the best result must reach.

    Returns
    -------
    bool
        True if evidence is sufficient, False if the query should be refused.
    """
    if not semantic_results:
        return False

    best_similarity = max(score for _, score in semantic_results)
    return best_similarity >= threshold
