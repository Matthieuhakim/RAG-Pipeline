from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
import numpy as np

from app.generation import embed_texts, generate_document_summary
from app.models import Chunk, ChunkMetadata

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
SUMMARIES_PATH = DATA_DIR / "summaries.json"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[tuple[int, str]] = []
    try:
        for index, page in enumerate(document, start=1):
            text = _normalize_whitespace(page.get_text("text"))
            if text:
                pages.append((index, text))
    finally:
        document.close()

    if not pages:
        raise ValueError("No extractable text was found in the PDF.")
    return pages


def chunk_pages(
    filename: str,
    pages: list[tuple[int, str]],
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    for page_number, text in pages:
        start = 0
        while start < len(text):
            end = _find_chunk_end(text, start, chunk_size)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            filename=filename,
                            page_number=page_number,
                            chunk_index=chunk_index,
                        ),
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break

            next_start = max(0, end - overlap)
            if next_start <= start:
                next_start = end
            start = next_start

    if not chunks:
        raise ValueError("The PDF did not produce any text chunks.")
    return chunks


async def ingest_pdf_bytes(
    filename: str,
    pdf_bytes: bytes,
) -> tuple[list[Chunk], np.ndarray, str]:
    pages = extract_pdf_pages(pdf_bytes)
    chunks = chunk_pages(filename, pages)
    embeddings = await embed_texts([chunk.text for chunk in chunks])
    summary = await generate_document_summary(chunks)
    return chunks, embeddings, summary


def save_store(chunks: list[Chunk], embeddings: np.ndarray, summaries: dict[str, str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    CHUNKS_PATH.write_text(
        json.dumps([chunk.model_dump() for chunk in chunks], indent=2),
        encoding="utf-8",
    )

    np.save(EMBEDDINGS_PATH, embeddings.astype(np.float32))
    SUMMARIES_PATH.write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def load_store() -> tuple[list[Chunk], np.ndarray, dict[str, str]]:
    chunks: list[Chunk] = []
    embeddings = np.empty((0, 0), dtype=np.float32)
    summaries: dict[str, str] = {}

    if CHUNKS_PATH.exists():
        raw_chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        chunks = [Chunk.model_validate(item) for item in raw_chunks]

    if EMBEDDINGS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = embeddings.astype(np.float32)

    if SUMMARIES_PATH.exists():
        summaries = json.loads(SUMMARIES_PATH.read_text(encoding="utf-8"))

    if chunks and len(chunks) != len(embeddings):
        raise RuntimeError(
            "Persisted store is inconsistent: chunk and embedding counts do not match."
        )

    if not chunks:
        embeddings = np.empty((0, 0), dtype=np.float32)

    return chunks, embeddings, summaries


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_chunk_end(text: str, start: int, chunk_size: int) -> int:
    ideal_end = min(len(text), start + chunk_size)
    if ideal_end >= len(text):
        return len(text)

    window = text[start:ideal_end]
    punctuation_candidates = [window.rfind(mark) for mark in (".", "!", "?")]
    boundary = max(punctuation_candidates)
    if boundary != -1 and boundary > int(chunk_size * 0.6):
        return start + boundary + 1

    whitespace_boundary = window.rfind(" ")
    if whitespace_boundary != -1 and whitespace_boundary > int(chunk_size * 0.6):
        return start + whitespace_boundary

    return ideal_end
