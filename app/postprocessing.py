from __future__ import annotations


def reciprocal_rank_fusion(
    semantic_results: list[tuple[int, float]],
    keyword_results: list[tuple[int, float]],
    *,
    k: int = 60,
    top_k: int = 6,
) -> list[tuple[int, float]]:
    fused_scores: dict[int, float] = {}

    for rank, (chunk_index, _) in enumerate(semantic_results, start=1):
        fused_scores[chunk_index] = fused_scores.get(chunk_index, 0.0) + 1.0 / (k + rank)

    for rank, (chunk_index, _) in enumerate(keyword_results, start=1):
        fused_scores[chunk_index] = fused_scores.get(chunk_index, 0.0) + 1.0 / (k + rank)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]
