from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from aisimplerag.app import create_app
from aisimplerag.api import router as router_module


class FakeQARecord:
    def __init__(self, qa_id: int, question: str, answer: str) -> None:
        now = datetime.now(timezone.utc)
        self.id = qa_id
        self.question = question
        self.answer = answer
        self.question_embedding = [0.1, 0.2]
        self.created_at = now
        self.updated_at = now


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("aisimplerag.app.init_db", lambda: None)
    app = create_app()

    def override_db():
        yield object()

    app.dependency_overrides[router_module.get_db] = override_db

    with TestClient(app) as test_client:
        yield test_client


def test_create_qa_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aisimplerag.api.router.generate_question_embedding", lambda _: [0.1, 0.2])
    monkeypatch.setattr(
        "aisimplerag.api.router.create_qa",
        lambda db, question, answer, question_embedding: FakeQARecord(1, question, answer),
    )

    response = client.post("/qa", json={"question": "q1", "answer": "a1"})

    assert response.status_code == 201
    payload = response.json()
    assert payload["id"] == 1
    assert payload["question"] == "q1"
    assert payload["answer"] == "a1"


def test_update_qa_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aisimplerag.api.router.update_qa_with_regenerated_embedding",
        lambda db, qa_id, question, answer: FakeQARecord(qa_id, question or "old-q", answer or "old-a"),
    )

    response = client.put("/qa/7", json={"question": "new-q"})

    assert response.status_code == 200
    assert response.json()["question"] == "new-q"


def test_delete_qa_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aisimplerag.api.router.delete_qa", lambda db, qa_id: qa_id == 2)

    response = client.delete("/qa/2")

    assert response.status_code == 204


def test_get_and_list_qa_endpoints(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aisimplerag.api.router.get_qa_by_id", lambda db, qa_id: FakeQARecord(qa_id, "q", "a"))
    monkeypatch.setattr(
        "aisimplerag.api.router.list_qa",
        lambda db, limit, offset: [FakeQARecord(1, "q1", "a1"), FakeQARecord(2, "q2", "a2")],
    )

    get_response = client.get("/qa/1")
    list_response = client.get("/qa")

    assert get_response.status_code == 200
    assert get_response.json()["id"] == 1
    assert get_response.json()["question_embedding"] == [0.1, 0.2]
    assert list_response.status_code == 200
    assert len(list_response.json()) == 2
