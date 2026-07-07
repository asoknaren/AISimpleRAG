# QA RAG Service Design Spec

## 1. Purpose
Build a simple FastAPI service that stores question-answer pairs in PostgreSQL and performs semantic search with pgvector. Questions are embedded with sentence-transformers and stored as vectors. A local Ollama model generates a final answer from retrieved matches using a retrieval-augmented generation flow.

## 2. Goals
- Provide CRUD operations for QA pairs.
- Support question search by semantic similarity.
- Return the top 5 matches when similarity is at least 0.85.
- Generate a final response with local Ollama using retrieved QA context.
- Keep the implementation small, local-first, and easy to run with `uv`.

## 3. Out of Scope
- Authentication and authorization.
- Multi-tenant support.
- Bulk import/export workflows.
- Document ingestion outside QA pair records.
- Remote LLM providers.

## 4. High-Level Architecture
The service is organized into four layers:
- API layer: FastAPI routers and request/response models.
- Service layer: embedding, retrieval, and RAG orchestration.
- Data layer: PostgreSQL access with pgvector support.
- Integration layer: sentence-transformers for embeddings and Ollama for generation.

## 5. Request Flow
### 5.1 CRUD flow
1. Client submits a QA record.
2. API validates the payload.
3. Service generates an embedding from the question text.
4. Data layer stores the QA record and vector.
5. API returns the persisted record.

### 5.2 Search flow
1. Client submits a question string.
2. Service converts the query into an embedding.
3. Data layer performs vector similarity search in PostgreSQL.
4. Results are filtered to cosine similarity greater than or equal to 0.85.
5. The top 5 matches are returned.

### 5.3 RAG flow
1. Client submits a question string.
2. Service performs the same vector search used for semantic lookup.
3. Retrieved QA pairs are passed to Ollama as context.
4. Ollama generates the final answer.
5. API returns the generated answer plus the retrieved context records.

## 6. Data Model
A QA record should contain:
- `id`
- `question`
- `answer`
- `question_embedding`
- `created_at`
- `updated_at`

Design rules:
- Only the question text is embedded.
- The answer remains plain text.
- The vector column is stored in pgvector.
- The embedding must be regenerated when the question changes.

## 7. API Surface
### 7.1 CRUD endpoints
- Create QA pair
- Update QA pair
- Delete QA pair
- Get QA pair by id
- List QA pairs

### 7.2 Search endpoint
- Input: raw question text
- Output: up to 5 matching QA pairs
- Rule: cosine similarity must be at least 0.85

### 7.3 RAG endpoint
- Input: raw question text
- Output: generated answer plus the retrieved QA context
- Rule: retrieve first, then generate from context

## 8. Configuration
Configuration should be environment-variable driven and grouped into three areas:
- Embedding config: model name, embedding dimension
- Ollama config: base URL, model name, generation settings
- PostgreSQL config: host, port, database, username, password, schema

Recommended config behavior:
- Provide local defaults for development.
- Keep sensitive values out of source control.
- Make the embedding model and Ollama model selectable without code changes.

## 9. Non-Functional Requirements
- Local development on macOS.
- Minimal dependencies.
- Clear separation between concerns.
- Observability for embedding, search, and Ollama calls.
- Deterministic retrieval behavior for testing.

## 10. Validation Criteria
The design is acceptable when:
- CRUD endpoints are fully specified.
- Search behavior is precise and testable.
- RAG flow is retrieval-first and generation-second.
- Configuration coverage is sufficient to run locally.
- The scope exclusions are clear enough to prevent feature drift.
