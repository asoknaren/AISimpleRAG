from collections.abc import Sequence

from aisimplerag.db.store import QAStore, StoredQARecord
from aisimplerag.services.embedding import generate_question_embedding


def create_qa(
    db: QAStore,
    *,
    question: str,
    answer: str,
    question_embedding: Sequence[float],
) -> StoredQARecord:
    return db.create_qa(question=question, answer=answer, question_embedding=question_embedding)


def get_qa_by_id(db: QAStore, qa_id: int) -> StoredQARecord | None:
    return db.get_qa_by_id(qa_id)


def list_qa(db: QAStore, *, limit: int = 100, offset: int = 0) -> list[StoredQARecord]:
    return db.list_qa(limit=limit, offset=offset)


def update_qa(
    db: QAStore,
    *,
    qa_id: int,
    question: str | None = None,
    answer: str | None = None,
    question_embedding: Sequence[float] | None = None,
) -> StoredQARecord | None:
    return db.update_qa(
        qa_id=qa_id,
        question=question,
        answer=answer,
        question_embedding=question_embedding,
    )


def update_qa_with_regenerated_embedding(
    db: QAStore,
    *,
    qa_id: int,
    question: str | None = None,
    answer: str | None = None,
) -> StoredQARecord | None:
    embedding: Sequence[float] | None = None
    if question is not None:
        embedding = generate_question_embedding(question)

    return update_qa(
        db,
        qa_id=qa_id,
        question=question,
        answer=answer,
        question_embedding=embedding,
    )


def delete_qa(db: QAStore, qa_id: int) -> bool:
    return db.delete_qa(qa_id)


def search_similar_questions(
    db: QAStore,
    *,
    query_embedding: Sequence[float],
    min_score: float | None = None,
    top_k: int | None = None,
) -> list[tuple[StoredQARecord, float]]:
    return db.search_similar_questions(
        query_embedding=query_embedding,
        min_score=min_score,
        top_k=top_k,
    )
