import pytest

from aisimplerag.services import retrieval


def test_retrieve_matches_uses_default_threshold_and_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr("aisimplerag.services.retrieval.generate_question_embedding", lambda question: [0.25, 0.75])

    def fake_search(db, query_embedding, min_score, top_k):
        captured["db"] = db
        captured["query_embedding"] = query_embedding
        captured["min_score"] = min_score
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr("aisimplerag.services.retrieval.search_similar_questions", fake_search)

    retrieval.retrieve_matches(object(), question="hello")

    assert captured["query_embedding"] == [0.25, 0.75]
    assert captured["min_score"] == 0.45
    assert captured["top_k"] == 5
