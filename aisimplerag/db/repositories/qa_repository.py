from collections.abc import Sequence

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from aisimplerag.core.config import settings
from aisimplerag.db.models.qa import QARecord
from aisimplerag.services.embedding import generate_question_embedding


def create_qa(
    db: Session,
    *,
    question: str,
    answer: str,
    question_embedding: Sequence[float],
) -> QARecord:
    record = QARecord(
        question=question,
        answer=answer,
        question_embedding=[float(value) for value in question_embedding],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_qa_by_id(db: Session, qa_id: int) -> QARecord | None:
    return db.get(QARecord, qa_id)


def list_qa(db: Session, *, limit: int = 100, offset: int = 0) -> list[QARecord]:
    stmt = (
        select(QARecord)
        .order_by(QARecord.id.asc())
        .offset(max(offset, 0))
        .limit(max(limit, 0))
    )
    return list(db.scalars(stmt).all())


def update_qa(
    db: Session,
    *,
    qa_id: int,
    question: str | None = None,
    answer: str | None = None,
    question_embedding: Sequence[float] | None = None,
) -> QARecord | None:
    record = get_qa_by_id(db, qa_id)
    if record is None:
        return None

    if question is not None:
        record.question = question
    if answer is not None:
        record.answer = answer
    if question_embedding is not None:
        record.question_embedding = [float(value) for value in question_embedding]

    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def update_qa_with_regenerated_embedding(
    db: Session,
    *,
    qa_id: int,
    question: str | None = None,
    answer: str | None = None,
) -> QARecord | None:
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


def delete_qa(db: Session, qa_id: int) -> bool:
    record = get_qa_by_id(db, qa_id)
    if record is None:
        return False

    db.delete(record)
    db.commit()
    return True


def search_similar_questions(
    db: Session,
    *,
    query_embedding: Sequence[float],
    min_score: float | None = None,
    top_k: int | None = None,
) -> list[tuple[QARecord, float]]:
    score_threshold = settings.search_min_score if min_score is None else min_score
    max_results = settings.search_top_k if top_k is None else top_k

    vector_query = [float(value) for value in query_embedding]
    distance_expr = QARecord.question_embedding.cosine_distance(vector_query)
    similarity_expr = (1.0 - distance_expr).label("similarity")

    stmt: Select[tuple[QARecord, float]] = (
        select(QARecord, similarity_expr)
        .where(similarity_expr >= score_threshold)
        .order_by(similarity_expr.desc(), QARecord.id.asc())
        .limit(max_results)
    )

    rows = db.execute(stmt).all()
    return [(record, float(similarity)) for record, similarity in rows]
