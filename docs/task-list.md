# QA RAG Service Task List

## Phase 0: Foundation
- [ ] Confirm the final spec and design doc are the source of truth.
- [ ] Decide the initial sentence-transformer model and Ollama model name.
- [ ] Decide the PostgreSQL schema and embedding dimension.
- [ ] Confirm the local development workflow uses `uv`.

## Phase 1: Project Setup
- [ ] Add core dependencies to `pyproject.toml`.
- [ ] Create a settings/config module for embedding, Ollama, and PostgreSQL values.
- [ ] Add a reusable logging setup.
- [ ] Create the package/module layout for API, services, and database code.

## Phase 2: Data Layer
- [ ] Define the QA table schema.
- [ ] Add pgvector support for the embedding column.
- [ ] Create database connection and session utilities.
- [ ] Implement insert, update, delete, and fetch operations for QA records.
- [ ] Implement vector search against the QA table.

## Phase 3: Embeddings and Retrieval
- [ ] Load the sentence-transformer model.
- [ ] Implement question embedding generation.
- [ ] Recompute embeddings when the question text changes.
- [ ] Implement cosine similarity filtering with a minimum score of 0.85.
- [ ] Return at most 5 matches from search.

## Phase 4: API Layer
- [ ] Implement QA create endpoint.
- [ ] Implement QA update endpoint.
- [ ] Implement QA delete endpoint.
- [ ] Implement QA get endpoint.
- [ ] Implement QA list endpoint.
- [ ] Implement question search endpoint.
- [ ] Implement RAG answer endpoint.
- [ ] Add request/response models and validation.

## Phase 5: Ollama Integration
- [ ] Add Ollama client/service integration.
- [ ] Build the retrieval-augmented prompt from matched QA records.
- [ ] Generate the final answer from the retrieved context.
- [ ] Return the generated output and supporting matches together.

## Phase 6: Testing
- [ ] Add tests for CRUD behavior.
- [ ] Add tests for similarity threshold logic.
- [ ] Add tests for top-5 search limits.
- [ ] Add tests for embedding regeneration on question updates.
- [ ] Add tests for RAG response structure.

## Phase 7: Documentation and Cleanup
- [ ] Update `README.md` with setup and run instructions.
- [ ] Document environment variables and defaults.
- [ ] Remove or replace any outdated scratch notes.
- [ ] Verify the project runs cleanly end to end.
