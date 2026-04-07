# RAG Pipeline

This project is a minimal retrieval-augmented generation pipeline built with FastAPI and a single-page HTML interface. It ingests PDF files, stores chunked text and embeddings on disk, performs hybrid retrieval with semantic similarity and BM25, and generates answers with Mistral AI.

```
Architecture Flow (ASCII):

  Upload PDFs
      │
      ▼
┌─────────────┐
│  Ingestion   │  extract text (PyMuPDF) → chunk (fixed-size + overlap)
│              │  → embed via Mistral → generate per-file summary via Mistral
│              │  → persist chunks.json + embeddings.npy + summaries.json
└──────────────┘

  User Query
      │
      ▼
┌─────────────┐
│   Query      │  intent detection (Mistral call w/ summaries as context)
│              │  → if no retrieval needed: answer directly, return
│              │  → if retrieval needed: transform query for better retrieval
└─────┬───────┘
      │
      ▼
┌─────────────┐
│   Search     │  semantic search (cosine similarity on embeddings)
│              │  + keyword search (BM25 from scratch)
└─────┬───────┘
      │
      ▼
┌──────────────┐
│ Postprocessing│  merge semantic + BM25 via Reciprocal Rank Fusion
│              │  → deduplicate → top-k
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Generation  │  prompt template + top-k chunks → Mistral → answer
└─────────────┘
```

## System Design

### Ingestion

PDF extraction uses PyMuPDF because it is fast, self-contained, and handles a wide range of native PDF layouts without requiring external tools like Poppler. The chunking strategy is fixed-size character windows with overlap, but chunk boundaries try to end on sentence punctuation first so the retrieved text remains more coherent. This keeps implementation simple while still reducing the chance that important context is cut off between adjacent chunks. A limitation of this approach is that scanned PDFs without embedded text will not be OCR'd.

### Query Processing

Each query first goes through intent detection using the document summaries as lightweight context. If retrieval is unnecessary, the app answers directly with Mistral. If retrieval is needed, the same model rewrites the user question into a more retrieval-friendly query, which helps both semantic embedding search and keyword matching land on better chunks.

### Search

Retrieval combines two signals: cosine similarity over stored embeddings and a BM25 implementation built from scratch over chunk text. Semantic search helps with paraphrases and concept matching, while BM25 catches exact wording, names, and domain-specific terms that embeddings can sometimes underweight. Using both gives more resilient retrieval than either method alone.

### Postprocessing

The two ranked lists are merged with Reciprocal Rank Fusion (RRF). RRF works well for hybrid retrieval because it rewards chunks that rank highly in either or both systems without needing manual score normalization between semantic and lexical signals. The result is a simple, stable fusion method that usually improves top-k quality.

### Generation

Answer generation uses a grounded prompt that injects the selected chunks with filename and page metadata. The system prompt tells the model to answer only from the supplied context and to admit when the context is insufficient. That reduces hallucination risk and makes source attribution straightforward in the UI response.

## How To Run

1. Install dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Ensure `.env` contains your Mistral API key:

   ```env
   MISTRAL_API_KEY=your_key_here
   ```

3. Start the server:

   ```bash
   uvicorn app.main:app --reload
   ```

4. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## API Documentation

### `POST /ingest`

Accepts one or more PDF files via multipart form data. Each file is validated, extracted, chunked, embedded, summarized, and merged into the persisted local store.

### `POST /query`

Accepts a JSON body with a `query` string. The backend decides whether retrieval is necessary, optionally performs hybrid search, and returns the generated answer plus source references and a retrieval flag.

### `GET /status`

Returns the number of ingested files, total chunks, and the filenames currently represented in the store.

### `GET /`

Serves the single-page chat and upload interface.

## Libraries Used

- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [NumPy](https://numpy.org/)
- [httpx](https://www.python-httpx.org/)

## Limitations And Future Improvements

- Scanned PDFs without embedded text are not supported because there is no OCR step.
- The chunking strategy is intentionally simple and character-based rather than token-aware.
- BM25 is fully in memory and rebuilt after updates, which is fine for small corpora but not large-scale deployments.
- Ingestion and querying are synchronous from the user's perspective; background jobs or streaming progress would improve UX.
- Re-ranking, metadata filters, and answer citation spans would be good next improvements.
