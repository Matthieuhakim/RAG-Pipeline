from __future__ import annotations

from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.generation import embed_texts, generate_grounded_answer
from app.ingestion import ingest_pdf_bytes, load_store, save_store
from app.models import ErrorResponse, IngestResponse, QueryRequest, QueryResponse, StatusResponse
from app.postprocessing import reciprocal_rank_fusion
from app.query import answer_directly, detect_intent, transform_query
from app.search import HybridSearchStore
from bonus.similarity_threshold import INSUFFICIENT_EVIDENCE_MESSAGE, check_similarity_threshold

ROOT_DIR = Path(__file__).resolve().parent.parent
UI_PATH = ROOT_DIR / "ui" / "index.html"

app = FastAPI(title="RAG Pipeline")
app.state.search_store = HybridSearchStore()


def error_response(
    status_code: int,
    error_type: str,
    message: str,
    details: object | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error_type=error_type,
        message=message,
        details=details,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


@app.on_event("startup")
async def startup_event() -> None:
    load_dotenv()
    chunks, embeddings, summaries = load_store()
    app.state.search_store.set_store(chunks, embeddings, summaries)


@app.get("/", include_in_schema=False)
async def serve_ui() -> FileResponse:
    return FileResponse(UI_PATH)


@app.get("/status", response_model=StatusResponse)
async def get_status(request: Request) -> StatusResponse:
    store: HybridSearchStore = request.app.state.search_store
    filenames = sorted(store.summaries.keys())
    return StatusResponse(
        ingested_files=len(filenames),
        total_chunks=len(store.chunks),
        filenames=filenames,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    request: Request,
    files: list[UploadFile] = File(...),
) -> IngestResponse | JSONResponse:
    if not files:
        return error_response(400, "validation_error", "No files were uploaded.")

    store: HybridSearchStore = request.app.state.search_store
    processed_filenames: list[str] = []
    errors: list[str] = []
    mistral_errors: list[str] = []
    new_chunks = []
    embedding_batches = []
    new_summaries: dict[str, str] = {}

    for file in files:
        filename = file.filename or "uploaded.pdf"
        if not filename.lower().endswith(".pdf"):
            errors.append(f"{filename}: only PDF files are supported.")
            continue

        try:
            content = await file.read()
            chunks, embeddings, summary = await ingest_pdf_bytes(filename, content)
        except ValueError as exc:
            errors.append(f"{filename}: {exc}")
            continue
        except RuntimeError as exc:
            message = f"{filename}: {exc}"
            errors.append(message)
            mistral_errors.append(message)
            continue
        except Exception as exc:
            errors.append(f"{filename}: failed to process file ({exc}).")
            continue

        new_chunks.extend(chunks)
        embedding_batches.append(embeddings)
        new_summaries[filename] = summary
        processed_filenames.append(filename)

    if processed_filenames:
        stacked_embeddings = (
            embedding_batches[0]
            if len(embedding_batches) == 1
            else np.vstack(embedding_batches)
        )
        store.upsert_documents(new_chunks, stacked_embeddings, new_summaries)
        save_store(store.chunks, store.embeddings, store.summaries)

    if not processed_filenames:
        if mistral_errors and len(mistral_errors) == len(errors):
            return error_response(
                502,
                "mistral_api_error",
                "A request to Mistral failed during ingestion.",
                details=errors,
            )
        return error_response(
            400,
            "ingestion_error",
            "No files were successfully ingested.",
            details=errors,
        )

    status = "success" if not errors else "partial_success"
    return IngestResponse(
        status=status,
        filenames_processed=processed_filenames,
        chunk_count=len(store.chunks),
        errors=errors,
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: Request,
    payload: QueryRequest,
) -> QueryResponse | JSONResponse:
    query = payload.query.strip()
    if not query:
        return error_response(400, "validation_error", "Query must not be empty.")

    store: HybridSearchStore = request.app.state.search_store

    try:
        if not store.has_data():
            answer = await answer_directly(query)
            return QueryResponse(answer=answer, sources=[], retrieval_used=False)

        decision = await detect_intent(query, store.summaries)
        if not decision.needs_retrieval:
            answer = await answer_directly(query)
            return QueryResponse(answer=answer, sources=[], retrieval_used=False)

        transformed_query = await transform_query(query)
        query_embedding = (await embed_texts([transformed_query]))[0]
        semantic_results, keyword_results = store.hybrid_search(
            transformed_query,
            query_embedding,
            top_k=8,
        )
        fused_results = reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            top_k=6,
        )

        if not fused_results or not check_similarity_threshold(semantic_results):
            return QueryResponse(
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                sources=[],
                retrieval_used=True,
            )

        selected_chunks = [store.chunks[index] for index, _ in fused_results]
        answer, sources = await generate_grounded_answer(query, selected_chunks)
        return QueryResponse(
            answer=answer,
            sources=sources,
            retrieval_used=True,
        )
    except RuntimeError as exc:
        return error_response(
            502,
            "mistral_api_error",
            "A request to Mistral failed.",
            details=str(exc),
        )
    except Exception as exc:
        return error_response(
            500,
            "query_error",
            "Failed to answer the query.",
            details=str(exc),
        )
