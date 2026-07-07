# QA RAG Service Task List

## Phase 0: Foundation
- [ ] Confirm the final spec and design doc are the source of truth.
- [ ] Decide the initial sentence-transformer model and Ollama model name.
- [ ] Decide the PostgreSQL schema and embedding dimension.
- [ ] Confirm the local development workflow uses `uv`.

## Phase 1: Project Setup
- [x] Add core dependencies to `pyproject.toml`.
- [x] Create a settings/config module for embedding, Ollama, and PostgreSQL values.
- [x] Add a reusable logging setup.
- [x] Create the package/module layout for API, services, and database code.

## Phase 2: Data Layer
- [x] Define the QA table schema.
- [x] Add pgvector support for the embedding column.
- [x] Create database connection and session utilities.
- [x] Implement insert, update, delete, and fetch operations for QA records.
- [x] Implement vector search against the QA table.

## Phase 3: Embeddings and Retrieval
- [x] Load the sentence-transformer model.
- [x] Implement question embedding generation.
- [x] Recompute embeddings when the question text changes.
- [x] Implement cosine similarity filtering with a minimum score of 0.85.
- [x] Return at most 5 matches from search.

## Phase 4: API Layer
- [x] Implement QA create endpoint.
- [x] Implement QA update endpoint.
- [x] Implement QA delete endpoint.
- [x] Implement QA get endpoint.
- [x] Implement QA list endpoint.
- [x] Implement question search endpoint.
- [x] Implement RAG answer endpoint.
- [x] Add request/response models and validation.

## Phase 5: Ollama Integration
- [x] Add Ollama client/service integration.
- [x] Build the retrieval-augmented prompt from matched QA records.
- [x] Generate the final answer from the retrieved context.
- [x] Return the generated output and supporting matches together.

## Phase 6: Testing
- [x] Add tests for CRUD behavior.
- [x] Add tests for similarity threshold logic.
- [x] Add tests for top-5 search limits.
- [x] Add tests for embedding regeneration on question updates.
- [x] Add tests for RAG response structure.

## Phase 7: Documentation and Cleanup
- [x] Update `README.md` with setup and run instructions.
- [x] Document environment variables and defaults.
- [x] Remove or replace any outdated scratch notes.
- [x] Verify the project runs cleanly end to end.
