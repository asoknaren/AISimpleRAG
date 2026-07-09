from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class QABase(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class QACreateRequest(QABase):
    pass


class QAUpdateRequest(BaseModel):
    question: str | None = Field(default=None, min_length=1)
    answer: str | None = Field(default=None, min_length=1)


class QAResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    answer: str
    created_at: datetime
    updated_at: datetime


class QADetailResponse(QAResponse):
    question_embedding: list[float]


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)


class SearchMatch(BaseModel):
    qa: QAResponse
    similarity: float


class SearchResponse(BaseModel):
    matches: list[SearchMatch]


class RAGRequest(BaseModel):
    question: str = Field(min_length=1)


class RAGResponse(BaseModel):
    question: str
    generated_answer: str
    matches: list[SearchMatch]
