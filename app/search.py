from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from app.models import Chunk


class HybridSearchStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.summaries: dict[str, str] = {}
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._document_frequencies: dict[str, int] = {}
        self._average_doc_length = 0.0

    def set_store(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        summaries: dict[str, str],
    ) -> None:
        self.chunks = list(chunks)
        self.embeddings = embeddings.astype(np.float32) if len(embeddings) else np.empty((0, 0), dtype=np.float32)
        self.summaries = dict(summaries)
        self.build_bm25_index()

    def upsert_documents(
        self,
        new_chunks: list[Chunk],
        new_embeddings: np.ndarray,
        new_summaries: dict[str, str],
    ) -> None:
        filenames = set(new_summaries.keys())
        retained_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.metadata.filename not in filenames
        ]

        retained_chunks = [self.chunks[index] for index in retained_indices]
        if retained_indices and len(self.embeddings):
            retained_embeddings = self.embeddings[retained_indices]
        else:
            retained_embeddings = np.empty((0, 0), dtype=np.float32)

        if retained_chunks and new_chunks:
            embeddings = np.vstack([retained_embeddings, new_embeddings])
        elif retained_chunks:
            embeddings = retained_embeddings
        elif new_chunks:
            embeddings = new_embeddings.astype(np.float32)
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)

        summaries = {
            filename: summary
            for filename, summary in self.summaries.items()
            if filename not in filenames
        }
        summaries.update(new_summaries)

        self.set_store(retained_chunks + new_chunks, embeddings, summaries)

    def build_bm25_index(self) -> None:
        self._doc_term_freqs = []
        self._doc_lengths = []
        document_frequencies: defaultdict[str, int] = defaultdict(int)

        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            term_freqs = Counter(tokens)
            self._doc_term_freqs.append(term_freqs)
            self._doc_lengths.append(len(tokens))
            for token in term_freqs:
                document_frequencies[token] += 1

        self._document_frequencies = dict(document_frequencies)
        if self._doc_lengths:
            self._average_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        else:
            self._average_doc_length = 0.0

    def has_data(self) -> bool:
        return bool(self.chunks) and len(self.embeddings) == len(self.chunks)

    def semantic_search(self, query_embedding: np.ndarray, top_k: int = 8) -> list[tuple[int, float]]:
        if not self.has_data():
            return []

        query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        doc_vectors = self.embeddings

        doc_norms = np.linalg.norm(doc_vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        safe_denominator = doc_norms * max(query_norm, 1e-12)
        similarities = np.dot(doc_vectors, query_vector.T).reshape(-1) / np.maximum(
            safe_denominator,
            1e-12,
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            (int(index), float(similarities[index]))
            for index in top_indices
            if similarities[index] > 0
        ]

    def keyword_search(self, query_text: str, top_k: int = 8) -> list[tuple[int, float]]:
        if not self.chunks:
            return []

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        scores = np.zeros(len(self.chunks), dtype=np.float32)
        total_docs = len(self.chunks)
        k1 = 1.5
        b = 0.75

        for doc_index, term_freqs in enumerate(self._doc_term_freqs):
            doc_length = self._doc_lengths[doc_index] if self._doc_lengths else 0
            score = 0.0
            for token in query_tokens:
                if token not in term_freqs:
                    continue
                document_frequency = self._document_frequencies.get(token, 0)
                if document_frequency == 0:
                    continue

                idf = math.log(
                    1
                    + (total_docs - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                term_frequency = term_freqs[token]
                norm = k1 * (
                    1 - b + b * doc_length / max(self._average_doc_length, 1.0)
                )
                score += idf * ((term_frequency * (k1 + 1)) / (term_frequency + norm))
            scores[doc_index] = score

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (int(index), float(scores[index]))
            for index in top_indices
            if scores[index] > 0
        ]

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 8,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        semantic_results = self.semantic_search(query_embedding, top_k=top_k)
        keyword_results = self.keyword_search(query_text, top_k=top_k)
        return semantic_results, keyword_results


def tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]
