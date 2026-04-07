from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    filename: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)


class Chunk(BaseModel):
    text: str = Field(min_length=1)
    metadata: ChunkMetadata


class SourceReference(BaseModel):
    filename: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    retrieval_used: bool


class IngestResponse(BaseModel):
    status: str
    filenames_processed: list[str] = Field(default_factory=list)
    chunk_count: int = Field(ge=0)
    errors: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error_type: str
    message: str
    details: Any | None = None


class StatusResponse(BaseModel):
    ingested_files: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    filenames: list[str] = Field(default_factory=list)


class IntentDecision(BaseModel):
    needs_retrieval: bool
    reasoning: str = ""
