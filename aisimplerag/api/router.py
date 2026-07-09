from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from aisimplerag.api.schemas import (
    QACreateRequest,
    QADetailResponse,
    QAResponse,
    QAUpdateRequest,
    RAGRequest,
    RAGResponse,
    SearchMatch,
    SearchRequest,
    SearchResponse,
)
from aisimplerag.db.repositories.qa_repository import (
    create_qa,
    delete_qa,
    get_qa_by_id,
    list_qa,
    update_qa_with_regenerated_embedding,
)
from aisimplerag.db.session import get_db
from aisimplerag.services.embedding import generate_question_embedding
from aisimplerag.services.rag import build_rag_response
from aisimplerag.services.retrieval import retrieve_matches

api_router = APIRouter()


@api_router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@api_router.post("/qa", response_model=QAResponse, status_code=status.HTTP_201_CREATED, tags=["qa"])
def create_qa_endpoint(payload: QACreateRequest, db: Session = Depends(get_db)) -> QAResponse:
    embedding = generate_question_embedding(payload.question)
    record = create_qa(
        db,
        question=payload.question,
        answer=payload.answer,
        question_embedding=embedding,
    )
    return QAResponse.model_validate(record)


@api_router.put("/qa/{qa_id}", response_model=QAResponse, tags=["qa"])
def update_qa_endpoint(
    qa_id: int,
    payload: QAUpdateRequest,
    db: Session = Depends(get_db),
) -> QAResponse:
    if payload.question is None and payload.answer is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one field is required for update.",
        )

    record = update_qa_with_regenerated_embedding(
        db,
        qa_id=qa_id,
        question=payload.question,
        answer=payload.answer,
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="QA record not found.")
    return QAResponse.model_validate(record)


@api_router.delete("/qa/{qa_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["qa"])
def delete_qa_endpoint(qa_id: int, db: Session = Depends(get_db)) -> Response:
    deleted = delete_qa(db, qa_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="QA record not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@api_router.get("/qa/{qa_id}", response_model=QADetailResponse, tags=["qa"])
def get_qa_endpoint(qa_id: int, db: Session = Depends(get_db)) -> QADetailResponse:
    record = get_qa_by_id(db, qa_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="QA record not found.")
    return QADetailResponse.model_validate(record)


@api_router.get("/qa", response_model=list[QAResponse], tags=["qa"])
def list_qa_endpoint(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[QAResponse]:
    records = list_qa(db, limit=limit, offset=offset)
    return [QAResponse.model_validate(record) for record in records]


@api_router.post("/qa/search", response_model=SearchResponse, tags=["search"])
def search_questions_endpoint(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    matches = retrieve_matches(db, question=payload.question)
    return SearchResponse(
        matches=[
            SearchMatch(qa=QAResponse.model_validate(record), similarity=similarity)
            for record, similarity in matches
        ]
    )


@api_router.post("/qa/rag", response_model=RAGResponse, tags=["rag"])
def rag_answer_endpoint(payload: RAGRequest, db: Session = Depends(get_db)) -> RAGResponse:
    return build_rag_response(db, question=payload.question)
