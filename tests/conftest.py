"""Shared fixtures for the RAG pipeline test suite."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import fitz
import numpy as np
import pytest

from app.models import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Three realistic chunks from two different files."""
    return [
        Chunk(
            text="Machine learning is a subset of artificial intelligence.",
            metadata=ChunkMetadata(filename="ml_intro.pdf", page_number=1, chunk_index=0),
        ),
        Chunk(
            text="Neural networks are inspired by biological neurons.",
            metadata=ChunkMetadata(filename="ml_intro.pdf", page_number=1, chunk_index=1),
        ),
        Chunk(
            text="Photosynthesis converts sunlight into chemical energy.",
            metadata=ChunkMetadata(filename="biology.pdf", page_number=1, chunk_index=0),
        ),
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Deterministic embeddings (3 x 4) matching sample_chunks."""
    rng = np.random.RandomState(42)
    emb = rng.randn(3, 4).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


@pytest.fixture
def sample_summaries() -> dict[str, str]:
    return {
        "ml_intro.pdf": "An introduction to machine learning concepts.",
        "biology.pdf": "Overview of photosynthesis in plants.",
    }


# ---------------------------------------------------------------------------
# PDF fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_pdf_bytes() -> bytes:
    """A valid single-page PDF with known text content."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    tw = fitz.TextWriter(page.rect)
    tw.append((10, 30), "Hello world. This is a test document.", fontsize=10)
    tw.write_text(page)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """A valid PDF with no text content (blank page)."""
    doc = fitz.open()
    doc.new_page(width=200, height=100)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# Mistral API mocking helpers
# ---------------------------------------------------------------------------


def make_chat_response(content: str) -> dict:
    """Build a fake Mistral chat completion response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }


def make_embedding_response(vectors: list[list[float]]) -> dict:
    """Build a fake Mistral embedding response."""
    return {
        "data": [
            {"index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ]
    }


class FakeHTTPResponse:
    """Minimal stand-in for an httpx.Response."""

    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data
        self.text = json.dumps(data)

    def json(self):
        return self._data
