"""Tests for app/query.py — intent detection, query transformation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.models import IntentDecision
from app.query import _parse_intent_response, answer_directly, detect_intent, transform_query


# ---------------------------------------------------------------------------
# _parse_intent_response
# ---------------------------------------------------------------------------


class TestParseIntentResponse:
    def test_valid_json(self):
        resp = '{"needs_retrieval": true, "reasoning": "Relevant query."}'
        result = _parse_intent_response(resp)
        assert result.needs_retrieval is True
        assert result.reasoning == "Relevant query."

    def test_json_with_markdown_fences(self):
        resp = '```json\n{"needs_retrieval": false, "reasoning": "Greeting."}\n```'
        result = _parse_intent_response(resp)
        assert result.needs_retrieval is False

    def test_json_embedded_in_text(self):
        resp = 'Here is the result: {"needs_retrieval": true, "reasoning": "test"} done.'
        result = _parse_intent_response(resp)
        assert result.needs_retrieval is True

    def test_totally_invalid_defaults_to_retrieval(self):
        result = _parse_intent_response("I don't know what JSON is")
        assert result.needs_retrieval is True

    def test_malformed_json_fields(self):
        resp = '{"wrong_field": true}'
        result = _parse_intent_response(resp)
        assert result.needs_retrieval is True

    def test_empty_string(self):
        result = _parse_intent_response("")
        assert result.needs_retrieval is True


# ---------------------------------------------------------------------------
# detect_intent
# ---------------------------------------------------------------------------


class TestDetectIntent:
    @pytest.mark.asyncio
    async def test_no_summaries_returns_no_retrieval(self):
        result = await detect_intent("hello", {})
        assert result.needs_retrieval is False

    @pytest.mark.asyncio
    async def test_with_summaries_calls_llm(self):
        mock = AsyncMock(return_value='{"needs_retrieval": true, "reasoning": "test"}')
        with patch("app.query.chat_completion", mock):
            summaries = {"doc.pdf": "A test document."}
            result = await detect_intent("What is this about?", summaries)
        assert result.needs_retrieval is True
        mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_returns_no_retrieval(self):
        mock = AsyncMock(return_value='{"needs_retrieval": false, "reasoning": "greeting"}')
        with patch("app.query.chat_completion", mock):
            result = await detect_intent("hello", {"doc.pdf": "A document."})
        assert result.needs_retrieval is False


# ---------------------------------------------------------------------------
# transform_query
# ---------------------------------------------------------------------------


class TestTransformQuery:
    @pytest.mark.asyncio
    async def test_returns_stripped_string(self):
        mock = AsyncMock(return_value="  optimized search query  ")
        with patch("app.query.chat_completion", mock):
            result = await transform_query("What is machine learning?")
        assert result == "optimized search query"

    @pytest.mark.asyncio
    async def test_calls_llm(self):
        mock = AsyncMock(return_value="rewritten query")
        with patch("app.query.chat_completion", mock):
            await transform_query("original question")
        mock.assert_called_once()


# ---------------------------------------------------------------------------
# answer_directly
# ---------------------------------------------------------------------------


class TestAnswerDirectly:
    @pytest.mark.asyncio
    async def test_returns_string(self):
        mock = AsyncMock(return_value="Hello! How can I help?")
        with patch("app.query.chat_completion", mock):
            result = await answer_directly("hello")
        assert result == "Hello! How can I help?"
