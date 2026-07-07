import pytest

from aisimplerag.db.repositories import qa_repository


def test_update_qa_regenerates_embedding_when_question_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("aisimplerag.db.repositories.qa_repository.generate_question_embedding", lambda question: [0.4, 0.6])

    def fake_update_qa(db, qa_id, question, answer, question_embedding):
        captured["question"] = question
        captured["answer"] = answer
        captured["embedding"] = question_embedding
        return object()

    monkeypatch.setattr("aisimplerag.db.repositories.qa_repository.update_qa", fake_update_qa)

    qa_repository.update_qa_with_regenerated_embedding(
        object(),
        qa_id=12,
        question="new question",
        answer="new answer",
    )

    assert captured["question"] == "new question"
    assert captured["answer"] == "new answer"
    assert captured["embedding"] == [0.4, 0.6]


def test_update_qa_does_not_regenerate_embedding_without_question(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fail_generate(question: str):
        raise AssertionError("Embedding generation should not run when question is unchanged")

    monkeypatch.setattr("aisimplerag.db.repositories.qa_repository.generate_question_embedding", fail_generate)

    def fake_update_qa(db, qa_id, question, answer, question_embedding):
        captured["embedding"] = question_embedding
        return object()

    monkeypatch.setattr("aisimplerag.db.repositories.qa_repository.update_qa", fake_update_qa)

    qa_repository.update_qa_with_regenerated_embedding(
        object(),
        qa_id=5,
        answer="updated answer",
    )

    assert captured["embedding"] is None
