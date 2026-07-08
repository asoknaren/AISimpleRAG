from aisimplerag.core.config import settings
from aisimplerag.db.store import QAStore, StoredQARecord
from aisimplerag.db.repositories.qa_repository import search_similar_questions
from aisimplerag.services.embedding import generate_question_embedding


def retrieval_limits() -> tuple[float, int]:
    return settings.search_min_score, settings.search_top_k


def retrieve_matches(db: QAStore, *, question: str) -> list[tuple[StoredQARecord, float]]:
    query_embedding = generate_question_embedding(question)
    min_score, top_k = retrieval_limits()
    return search_similar_questions(
        db,
        query_embedding=query_embedding,
        min_score=min_score,
        top_k=top_k,
    )

