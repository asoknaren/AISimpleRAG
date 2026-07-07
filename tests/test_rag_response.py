from datetime import datetime, timezone

import pytest

from aisimplerag.services.rag import build_rag_response


class FakeQARecord:
    def __init__(self, qa_id: int, question: str, answer: str) -> None:
        now = datetime.now(timezone.utc)
        self.id = qa_id
        self.question = question
        self.answer = answer
        self.created_at = now
        self.updated_at = now


def test_build_rag_response_includes_generated_answer_and_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_matches = [
        (FakeQARecord(1, "stored q1", "stored a1"), 0.92),
        (FakeQARecord(2, "stored q2", "stored a2"), 0.88),
    ]

    monkeypatch.setattr("aisimplerag.services.rag.retrieve_matches", lambda db, question: fake_matches)
    monkeypatch.setattr("aisimplerag.services.rag.generate_with_ollama", lambda prompt: "final answer")

    response = build_rag_response(object(), question="user question")

    assert response.question == "user question"
    assert response.generated_answer == "final answer"
    assert len(response.matches) == 2
    assert response.matches[0].qa.id == 1
    assert response.matches[0].similarity == 0.92
