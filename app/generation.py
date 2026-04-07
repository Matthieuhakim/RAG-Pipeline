from __future__ import annotations

import os
from typing import Any, Sequence

import httpx
import numpy as np

from app.models import Chunk, SourceReference

MISTRAL_API_BASE_URL = "https://api.mistral.ai/v1"
CHAT_MODEL = "mistral-small-latest"
EMBEDDING_MODEL = "mistral-embed"


def _get_api_key() -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY in the environment.")
    return api_key


def _build_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


async def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str = CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{MISTRAL_API_BASE_URL}/chat/completions",
            headers=_build_headers(),
            json=payload,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Mistral chat request failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Mistral chat response did not include any choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        content = "\n".join(part for part in text_parts if part).strip()

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Mistral chat response did not include text content.")

    return content.strip()


async def embed_texts(texts: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    batches: list[np.ndarray] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            response = await client.post(
                f"{MISTRAL_API_BASE_URL}/embeddings",
                headers=_build_headers(),
                json={"model": EMBEDDING_MODEL, "input": batch},
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Mistral embeddings request failed ({response.status_code}): "
                    f"{response.text}"
                )

            data = response.json().get("data", [])
            ordered_rows = sorted(data, key=lambda item: item.get("index", 0))
            vectors = [row.get("embedding", []) for row in ordered_rows]
            if not vectors or any(not vector for vector in vectors):
                raise RuntimeError("Mistral embeddings response was empty or invalid.")
            batches.append(np.asarray(vectors, dtype=np.float32))

    return np.vstack(batches)


async def generate_document_summary(chunks: Sequence[Chunk], *, sample_size: int = 4) -> str:
    if not chunks:
        return "No summary available."

    sample = "\n\n".join(chunk.text for chunk in chunks[:sample_size])
    messages = [
        {
            "role": "system",
            "content": (
                "You summarize documents for retrieval systems. "
                "Write 2-3 sentences describing the document's main topics."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize this document in 2-3 sentences. "
                "What topics does it cover?\n\n"
                f"{sample}"
            ),
        },
    ]
    return await chat_completion(messages, temperature=0.1, max_tokens=180)


async def generate_grounded_answer(
    query: str,
    chunks: Sequence[Chunk],
) -> tuple[str, list[SourceReference]]:
    context_sections = []
    seen_sources: set[tuple[str, int, int]] = set()
    sources: list[SourceReference] = []

    for chunk in chunks:
        metadata = chunk.metadata
        context_sections.append(
            f"[Source: {metadata.filename}, page {metadata.page_number}, "
            f"chunk {metadata.chunk_index}]\n{chunk.text}"
        )

        source_key = (
            metadata.filename,
            metadata.page_number,
            metadata.chunk_index,
        )
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(
                SourceReference(
                    filename=metadata.filename,
                    page_number=metadata.page_number,
                    chunk_index=metadata.chunk_index,
                )
            )

    context = "\n\n".join(context_sections) if context_sections else "No context provided."
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question based ONLY "
                "on the provided context. If the context does not contain enough "
                "information, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n\n{context}\n\nQuestion: {query}"
            ),
        },
    ]
    answer = await chat_completion(messages, temperature=0.2, max_tokens=500)
    return answer, sources
