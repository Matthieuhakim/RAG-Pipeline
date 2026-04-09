"""Tests for app/generation.py — Mistral API wrappers."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.generation import (
    _DEFAULT_SYSTEM_PROMPT,
    _build_headers,
    _get_api_key,
    chat_completion,
    embed_texts,
    generate_document_summary,
    generate_grounded_answer,
)
from app.models import Chunk, ChunkMetadata
from tests.conftest import FakeHTTPResponse, make_chat_response, make_embedding_response


# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_returns_key(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key-123")
        assert _get_api_key() == "test-key-123"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="Missing MISTRAL_API_KEY"):
            _get_api_key()


class TestBuildHeaders:
    def test_includes_bearer_token(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "my-key")
        headers = _build_headers()
        assert headers["Authorization"] == "Bearer my-key"
        assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# chat_completion
# ---------------------------------------------------------------------------


class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_successful_response(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test")
        fake_resp = FakeHTTPResponse(200, make_chat_response("Hello!"))

        mock_post = AsyncMock(return_value=fake_resp)
        with patch("app.generation.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(post=mock_post))
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await chat_completion([{"role": "user", "content": "hi"}])

        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_http_error_raises(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test")
        fake_resp = FakeHTTPResponse(500, {"error": "server error"})

        mock_post = AsyncMock(return_value=fake_resp)
        with patch("app.generation.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(post=mock_post))
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="Mistral chat request failed"):
                await chat_completion([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_empty_choices_raises(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test")
        fake_resp = FakeHTTPResponse(200, {"choices": []})

        mock_post = AsyncMock(return_value=fake_resp)
        with patch("app.generation.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(post=mock_post))
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="did not include any choices"):
                await chat_completion([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_list_content_format(self, monkeypatch):
        """Mistral sometimes returns content as a list of objects."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test")
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Part one."},
                            {"type": "text", "text": "Part two."},
                        ],
                    }
                }
            ]
        }
        fake_resp = FakeHTTPResponse(200, data)

        mock_post = AsyncMock(return_value=fake_resp)
        with patch("app.generation.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=AsyncMock(post=mock_post))
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await chat_completion([{"role": "user", "content": "hi"}])

        assert "Part one." in result
        assert "Part two." in result


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------


class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_empty_input(self):
        result = await embed_texts([])
        assert result.shape == (0, 0)

    @pytest.mark.asyncio
    async def test_single_batch(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test")
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        fake_resp = FakeHTTPResponse(200, make_embedding_response(vectors))

        mock_post = AsyncMock(return_value=fake_resp)
        with patch("app.generation.httpx.AsyncClient") as MockClient:
            instance = AsyncMock(post=mock_post)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await embed_texts(["text1", "text2"])

        assert result.shape == (2, 3)
        assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_http_error_raises(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test")
        fake_resp = FakeHTTPResponse(429, {"error": "rate limited"})

        mock_post = AsyncMock(return_value=fake_resp)
        with patch("app.generation.httpx.AsyncClient") as MockClient:
            instance = AsyncMock(post=mock_post)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError, match="Mistral embeddings request failed"):
                await embed_texts(["text"])


# ---------------------------------------------------------------------------
# generate_document_summary
# ---------------------------------------------------------------------------


class TestGenerateDocumentSummary:
    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        result = await generate_document_summary([])
        assert result == "No summary available."

    @pytest.mark.asyncio
    async def test_with_chunks(self):
        mock = AsyncMock(return_value="A summary of the document.")
        chunks = [
            Chunk(text="content", metadata=ChunkMetadata(filename="f.pdf", page_number=1, chunk_index=0)),
        ]
        with patch("app.generation.chat_completion", mock):
            result = await generate_document_summary(chunks)
        assert result == "A summary of the document."


# ---------------------------------------------------------------------------
# generate_grounded_answer
# ---------------------------------------------------------------------------


class TestGenerateGroundedAnswer:
    @pytest.mark.asyncio
    async def test_returns_answer_and_sources(self):
        mock = AsyncMock(return_value="The answer is 42.")
        chunks = [
            Chunk(text="text1", metadata=ChunkMetadata(filename="a.pdf", page_number=1, chunk_index=0)),
            Chunk(text="text2", metadata=ChunkMetadata(filename="a.pdf", page_number=2, chunk_index=1)),
        ]
        with patch("app.generation.chat_completion", mock):
            answer, sources = await generate_grounded_answer("question", chunks)
        assert answer == "The answer is 42."
        assert len(sources) == 2

    @pytest.mark.asyncio
    async def test_deduplicates_sources(self):
        mock = AsyncMock(return_value="Answer.")
        meta = ChunkMetadata(filename="a.pdf", page_number=1, chunk_index=0)
        chunks = [
            Chunk(text="text1", metadata=meta),
            Chunk(text="text2", metadata=meta),
        ]
        with patch("app.generation.chat_completion", mock):
            _, sources = await generate_grounded_answer("q", chunks)
        assert len(sources) == 1

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self):
        mock = AsyncMock(return_value="Answer.")
        chunks = [
            Chunk(text="text", metadata=ChunkMetadata(filename="a.pdf", page_number=1, chunk_index=0)),
        ]
        with patch("app.generation.chat_completion", mock):
            await generate_grounded_answer("q", chunks, system_prompt="Custom prompt.")
        messages = mock.call_args[0][0]
        assert messages[0]["content"] == "Custom prompt."

    def test_default_system_prompt_exists(self):
        assert "ONLY" in _DEFAULT_SYSTEM_PROMPT
