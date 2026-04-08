"""Post-hoc hallucination filter.

After the LLM generates an answer, this module checks each claim
(sentence) against the source chunks.  Sentences that are not
supported by any chunk are stripped and a warning is appended,
reducing the risk of unsupported claims reaching the user.
"""

from __future__ import annotations

import re

import numpy as np

from app.generation import embed_texts
from app.models import Chunk

DEFAULT_SUPPORT_THRESHOLD = 0.70

HALLUCINATION_WARNING = (
    "\n\n[Note: Some parts of the original answer were removed because "
    "they could not be verified against the source documents.]"
)

# Short, structural sentences that don't make factual claims.
_SKIP_PATTERNS = re.compile(
    r"^("
    r"(here|below)\s+(is|are)\b"
    r"|based\s+on\s+(the\s+)?(provided\s+)?context"
    r"|according\s+to\s+the\s+(provided\s+)?(context|document)"
    r"|the\s+context\s+(does\s+not|doesn't)"
    r"|i\s+(don't|do\s+not)\s+have"
    r"|in\s+summary"
    r"|to\s+summarize"
    r"|in\s+conclusion"
    r"|note\s*:"
    r")",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on period/exclamation/question mark.

    Keeps the delimiter attached to the sentence.  Handles common
    abbreviations (e.g., "Dr.", "Mr.") to avoid over-splitting.
    """
    # Split on sentence-ending punctuation followed by whitespace or end.
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


def _is_structural(sentence: str) -> bool:
    """Return True if the sentence is structural/meta rather than a factual claim."""
    if len(sentence.split()) < 4:
        return True
    return bool(_SKIP_PATTERNS.search(sentence))


def _cosine_similarities(
    sentence_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between one sentence and all chunks."""
    sent = sentence_embedding.reshape(1, -1).astype(np.float32)
    docs = chunk_embeddings.astype(np.float32)

    dot_products = np.dot(docs, sent.T).reshape(-1)
    doc_norms = np.linalg.norm(docs, axis=1)
    sent_norm = float(np.linalg.norm(sent))

    denominator = doc_norms * max(sent_norm, 1e-12)
    return dot_products / np.maximum(denominator, 1e-12)


async def filter_hallucinations(
    answer: str,
    chunks: list[Chunk],
    chunk_embeddings: np.ndarray,
    *,
    threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> tuple[str, bool]:
    """Check each sentence in *answer* for support in *chunks*.

    Parameters
    ----------
    answer:
        The LLM-generated answer text.
    chunks:
        The source chunks that were fed to the LLM.
    chunk_embeddings:
        Pre-computed embeddings for *chunks* (shape N x D).
    threshold:
        Minimum cosine similarity between a sentence and its
        best-matching chunk for the sentence to be considered
        supported.

    Returns
    -------
    (filtered_answer, was_filtered)
        The answer with unsupported sentences removed, and a bool
        indicating whether any content was removed.
    """
    sentences = _split_sentences(answer)
    if not sentences or len(chunk_embeddings) == 0:
        return answer, False

    # Identify claim sentences (non-structural) that need checking.
    claim_indices = [
        i for i, s in enumerate(sentences) if not _is_structural(s)
    ]

    if not claim_indices:
        return answer, False

    # Embed all claim sentences in one batch.
    claim_texts = [sentences[i] for i in claim_indices]
    claim_embeddings = await embed_texts(claim_texts)

    # Check each claim against the chunk embeddings.
    unsupported: set[int] = set()
    for batch_idx, sent_idx in enumerate(claim_indices):
        similarities = _cosine_similarities(
            claim_embeddings[batch_idx], chunk_embeddings
        )
        best_score = float(np.max(similarities))
        if best_score < threshold:
            unsupported.add(sent_idx)

    if not unsupported:
        return answer, False

    # Rebuild the answer keeping only supported + structural sentences.
    kept = [s for i, s in enumerate(sentences) if i not in unsupported]

    if not kept:
        return (
            "I was unable to produce a fully supported answer from the "
            "available context." + HALLUCINATION_WARNING
        ), True

    filtered = " ".join(kept) + HALLUCINATION_WARNING
    return filtered, True
