# AISimpleRAG

Simple local-first QA RAG service built with FastAPI, PostgreSQL + pgvector, sentence-transformers, and Ollama.

## Features

- CRUD operations for QA pairs.
- Semantic search with cosine similarity thresholding.
- Top-k retrieval (default 5 matches).
- Retrieval-augmented answer generation through local Ollama.
- Test suite for CRUD, retrieval constraints, embedding regeneration, and RAG response shape.

## Prerequisites

- Python 3.10+
- `uv`
- PostgreSQL 14+ with `pgvector` extension available
- Ollama running locally (default: `http://localhost:11434`)

## Quick Start

1. Install dependencies:

```powershell
uv sync
```

2. Create a `.env` file (or copy from `.env.example`) and adjust values as needed.

3. Ensure PostgreSQL is available and has `pgvector` support.
CREATE EXTENSION IF NOT EXISTS vector;

using postgreSQL 17 version for testing

or use docker - 
docker run --name simplerag-db  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres$#$ -e POSTGRES_DB=aisimpleragtest -p 5434:5432 -v simplerag_data:/var/lib/postgresql/data -d pgvector/pgvector:pg17

4. Start Ollama and pull a model if needed:

```powershell
ollama pull llama3.1:8b
```

5. Run the API:

```powershell
uv run python main.py
```

API default URL: `http://127.0.0.1:8000`

## Environment Variables

The application is configured via environment variables (also documented in `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformer model used to embed question text |
| `EMBEDDING_DIMENSION` | `384` | Vector dimension stored in pgvector |
| `SEARCH_MIN_SCORE` | `0.85` | Minimum cosine similarity score for search results |
| `SEARCH_TOP_K` | `5` | Maximum number of retrieved matches |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server base URL |
| `OLLAMA_MODEL_NAME` | `llama3.1:8b` | Ollama model used for generation |
| `OLLAMA_TIMEOUT_SECONDS` | `30.0` | HTTP timeout for Ollama requests |
| `POSTGRES_HOST` | `127.0.0.1` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `aisimplerag` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `postgres` | Database password |
| `POSTGRES_SCHEMA` | `public` | Schema for QA table |

## API Endpoints

- `GET /health`
- `POST /qa`
- `PUT /qa/{qa_id}`
- `DELETE /qa/{qa_id}`
- `GET /qa/{qa_id}`
- `GET /qa`
- `POST /qa/search`
- `POST /qa/rag`

## Testing

Run all tests:

```powershell
uv run pytest -q
```

## Notes

- On startup, the app attempts to create the `vector` extension (`CREATE EXTENSION IF NOT EXISTS vector`) and creates schema/tables if missing.
- If Ollama is unavailable, the RAG service returns a deterministic fallback answer based on retrieved context.
