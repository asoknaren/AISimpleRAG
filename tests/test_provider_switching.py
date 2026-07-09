from datetime import datetime, timezone

import pytest

from aisimplerag.core.config import settings
from aisimplerag.services.embedding import generate_question_embedding
from aisimplerag.services.rag import build_rag_response


class FakeQARecord:
    def __init__(self, qa_id: int, question: str, answer: str) -> None:
        now = datetime.now(timezone.utc)
        self.id = qa_id
        self.question = question
        self.answer = answer
        self.created_at = now
        self.updated_at = now


def test_embedding_uses_openai_when_provider_is_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ai_provider", "openai")
    monkeypatch.setattr(settings, "embedding_dimension", 2)
    monkeypatch.setattr(
        "aisimplerag.services.embedding.generate_openai_embedding",
        lambda question: [0.12, 0.34],
    )

    def _fail_local_model_load():
        raise AssertionError("Local sentence-transformer should not load in OpenAI mode")

    monkeypatch.setattr("aisimplerag.services.embedding._load_sentence_transformer", _fail_local_model_load)

    embedding = generate_question_embedding("What is RAG?")

    assert embedding == [0.12, 0.34]


def test_rag_generation_uses_openai_when_provider_is_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ai_provider", "openai")
    fake_matches = [(FakeQARecord(1, "stored q1", "stored a1"), 0.9)]

    monkeypatch.setattr("aisimplerag.services.rag.retrieve_matches", lambda db, question: fake_matches)
    monkeypatch.setattr("aisimplerag.services.rag.generate_with_openai", lambda prompt: "openai answer")

    def _fail_ollama(prompt: str) -> str:
        raise AssertionError("Ollama path should not run in OpenAI mode")

    monkeypatch.setattr("aisimplerag.services.rag.generate_with_ollama", _fail_ollama)

    response = build_rag_response(object(), question="user question")

    assert response.generated_answer == "openai answer"
    assert response.matches[0].qa.id == 1