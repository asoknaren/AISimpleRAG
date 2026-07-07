# FastAPI QA RAG Service Specification

## 1. Overview
This service provides CRUD and semantic search for question-answer pairs using FastAPI, PostgreSQL, and pgvector. Question text is embedded with sentence-transformers and stored in the database. A local Ollama model is used for retrieval-augmented generation over the top matching QA pairs.

## 2. Scope
The system supports:
- Insert QA pairs
- Update QA pairs
- Delete QA pairs
- Select/list QA pairs
- Search by question text
- Generate a RAG answer from retrieved QA pairs

The system does not include:
- Authentication
- Multi-tenant support
- Bulk import/export
- Document ingestion beyond QA pair records

## 3. Functional Requirements
- Store QA pairs in PostgreSQL.
- Store the question embedding in pgvector.
- Generate embeddings for question text using sentence-transformers.
- Recompute the embedding whenever the question changes.
- Search by question text using cosine similarity.
- Return the top 5 matches with similarity greater than or equal to 0.85.
- Use local Ollama to generate the final answer from the retrieved context.
- Return both the generated response and the matched QA records.

## 4. Data Model
A QA record should include:
- id
- question
- answer
- question_embedding
- created_at
- updated_at

Recommended behavior:
- The embedding is generated from the question field only.
- The answer field is stored as plain text.
- The embedding column uses pgvector and is indexed for similarity search.
- Updates to the question must update the embedding.

## 5. API Requirements
Required endpoints:
- Create QA pair
- Update QA pair
- Delete QA pair
- Get QA pair by id
- List QA pairs
- Search QA pairs by question text
- Generate RAG answer from a question

Search endpoint rules:
- Accept raw question text as input
- Convert the query to an embedding
- Compare against stored question embeddings using cosine similarity
- Return only matches with score at or above 0.85
- Return at most 5 results

RAG endpoint rules:
- Use the same search flow to retrieve top matches
- Pass retrieved context to Ollama
- Return the generated answer plus the retrieved matches used as context

## 6. Configuration Requirements
The project must support configuration for:
- Embedding model name
- Embedding dimension
- Ollama base URL
- Ollama model name
- Ollama generation settings such as temperature and max tokens
- PostgreSQL host
- PostgreSQL port
- PostgreSQL database name
- PostgreSQL username
- PostgreSQL password
- PostgreSQL schema

Configuration should be environment-variable driven, with sensible local defaults.

## 7. Non-Functional Requirements
- Designed for local development on macOS
- Minimal dependency footprint
- Clear separation between API, embedding, database, and generation layers
- Logging for search, embedding, and Ollama calls
- Deterministic similarity threshold behavior for testing

## 8. Verification
The implementation is complete only if:
- CRUD endpoints work correctly
- Search returns only records at or above the threshold
- Search returns no more than 5 matches
- Question updates regenerate embeddings
- Ollama generation works with retrieved context
- Configuration can be loaded from environment variables
