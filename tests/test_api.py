"""End-to-end tests for the FastAPI application (app/main.py).

All Mistral API calls are mocked. Uses httpx.AsyncClient with ASGITransport
to test the full request/response cycle.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import pytest_asyncio
import httpx

from app.main import app
from app.models import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client():
    """Async test client that mocks startup (no disk load).

    Resets the search store to empty before each test to prevent state
    leaking between tests.
    """
    from app.search import HybridSearchStore

    with (
        patch("app.main.load_dotenv"),
        patch("app.main.load_store", return_value=([], np.empty((0, 0), dtype=np.float32), {})),
    ):
        # Reset the global store to empty for each test
        app.state.search_store = HybridSearchStore()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture
def tiny_pdf() -> bytes:
    """Minimal valid PDF bytes."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    tw = fitz.TextWriter(page.rect)
    tw.append((10, 30), "Solar panels convert sunlight into electricity.", fontsize=10)
    tw.write_text(page)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestServeUI:
    @pytest.mark.asyncio
    async def test_serves_html(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------


class TestStatus:
    @pytest.mark.asyncio
    async def test_empty_status(self, client):
        resp = await client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ingested_files"] == 0
        assert data["total_chunks"] == 0
        assert data["filenames"] == []


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


class TestIngest:
    @pytest.mark.asyncio
    async def test_no_files_returns_422(self, client):
        """FastAPI returns 422 when required multipart field is missing."""
        resp = await client.post("/ingest", files=[])
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_non_pdf_rejected(self, client):
        resp = await client.post(
            "/ingest",
            files=[("files", ("test.txt", b"hello", "text/plain"))],
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "only PDF" in data.get("details", [""])[0]

    @pytest.mark.asyncio
    async def test_valid_pdf_ingested(self, client, tiny_pdf):
        """Ingest a valid PDF with mocked Mistral calls."""
        fake_emb = np.random.randn(1, 4).astype(np.float32)

        async def mock_embed(texts, **kwargs):
            n = len(texts)
            return np.random.randn(n, 4).astype(np.float32)

        async def mock_chat(messages, **kwargs):
            return "A summary of solar panels."

        with (
            patch("app.ingestion.embed_texts", side_effect=mock_embed),
            patch("app.ingestion.generate_document_summary", side_effect=mock_chat),
            patch("app.main.save_store"),
        ):
            resp = await client.post(
                "/ingest",
                files=[("files", ("test.pdf", tiny_pdf, "application/pdf"))],
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "test.pdf" in data["filenames_processed"]
        assert data["chunk_count"] >= 1


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


class TestQuery:
    @pytest.mark.asyncio
    async def test_empty_query_returns_400(self, client):
        resp = await client.post("/query", json={"query": " "})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_pii_refused(self, client):
        resp = await client.post("/query", json={"query": "My email is a@b.com"})
        assert resp.status_code == 400
        data = resp.json()
        assert data["error_type"] == "query_refused"

    @pytest.mark.asyncio
    async def test_legal_refused(self, client):
        resp = await client.post("/query", json={"query": "Can I sue my boss?"})
        assert resp.status_code == 400
        assert resp.json()["error_type"] == "query_refused"

    @pytest.mark.asyncio
    async def test_medical_refused(self, client):
        resp = await client.post("/query", json={"query": "Should I take aspirin?"})
        assert resp.status_code == 400
        assert resp.json()["error_type"] == "query_refused"

    @pytest.mark.asyncio
    async def test_no_data_answers_directly(self, client):
        """With empty store, query should answer directly without retrieval."""
        mock = AsyncMock(return_value="Hello there!")
        with patch("app.main.answer_directly", mock):
            resp = await client.post("/query", json={"query": "hello"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval_used"] is False
        assert data["answer"] == "Hello there!"

    @pytest.mark.asyncio
    async def test_retrieval_query_full_pipeline(self, client, tiny_pdf):
        """Full pipeline: ingest, then query with retrieval."""
        # 1. Ingest a PDF
        async def mock_embed(texts, **kwargs):
            n = len(texts)
            rng = np.random.RandomState(42)
            return rng.randn(n, 4).astype(np.float32)

        async def mock_summary(chunks, **kwargs):
            return "Solar panel information."

        with (
            patch("app.ingestion.embed_texts", side_effect=mock_embed),
            patch("app.ingestion.generate_document_summary", side_effect=mock_summary),
            patch("app.main.save_store"),
        ):
            resp = await client.post(
                "/ingest",
                files=[("files", ("solar.pdf", tiny_pdf, "application/pdf"))],
            )
        assert resp.status_code == 200

        # 2. Query with all LLM calls mocked
        async def mock_chat(messages, **kwargs):
            # Detect which call this is based on system message content
            sys_msg = messages[0].get("content", "") if messages else ""
            if "decide whether" in sys_msg:
                return '{"needs_retrieval": true, "reasoning": "relevant"}'
            elif "rewrite" in sys_msg.lower() or "search" in sys_msg.lower():
                return "solar panel efficiency"
            else:
                return "Solar panels are 20% efficient."

        async def mock_embed_query(texts, **kwargs):
            n = len(texts)
            rng = np.random.RandomState(42)
            return rng.randn(n, 4).astype(np.float32)

        async def mock_filter(answer, chunks, embs, **kwargs):
            return answer, False

        with (
            patch("app.query.chat_completion", side_effect=mock_chat),
            patch("app.main.embed_texts", side_effect=mock_embed_query),
            patch("app.main.generate_grounded_answer", return_value=("Solar panels are 20% efficient.", [])),
            patch("app.main.filter_hallucinations", side_effect=mock_filter),
        ):
            resp = await client.post("/query", json={"query": "What is solar panel efficiency?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval_used"] is True
        assert "solar" in data["answer"].lower() or "20%" in data["answer"]


# ---------------------------------------------------------------------------
# Status after ingestion
# ---------------------------------------------------------------------------


class TestStatusAfterIngestion:
    @pytest.mark.asyncio
    async def test_status_reflects_ingestion(self, client, tiny_pdf):
        async def mock_embed(texts, **kwargs):
            return np.random.randn(len(texts), 4).astype(np.float32)

        async def mock_summary(chunks, **kwargs):
            return "Summary."

        with (
            patch("app.ingestion.embed_texts", side_effect=mock_embed),
            patch("app.ingestion.generate_document_summary", side_effect=mock_summary),
            patch("app.main.save_store"),
        ):
            await client.post(
                "/ingest",
                files=[("files", ("doc.pdf", tiny_pdf, "application/pdf"))],
            )

        resp = await client.get("/status")
        data = resp.json()
        assert data["ingested_files"] >= 1
        assert data["total_chunks"] >= 1
        assert "doc.pdf" in data["filenames"]
