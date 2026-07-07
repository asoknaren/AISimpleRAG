from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from types import SimpleNamespace

import pytest

from aisimplerag.core.config import settings
from aisimplerag.db import store as store_module
from aisimplerag.db.store import ChromaQAStore, PostgresQAStore


def _cosine_similarity(left: list[float], right: list[float]) -> float:
	left_norm = sqrt(sum(value * value for value in left))
	right_norm = sqrt(sum(value * value for value in right))
	if left_norm == 0 or right_norm == 0:
		return 0.0
	return sum(left[index] * right[index] for index in range(len(left))) / (left_norm * right_norm)


@dataclass
class _FakeRecord:
	id: str
	document: str
	metadata: dict[str, object]
	embedding: list[float]


class _FakeCollection:
	def __init__(self) -> None:
		self.records: dict[str, _FakeRecord] = {}

	def count(self) -> int:
		return len(self.records)

	def add(self, *, ids, documents, metadatas, embeddings) -> None:
		for qa_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings, strict=False):
			self.records[str(qa_id)] = _FakeRecord(
				id=str(qa_id),
				document=document,
				metadata=dict(metadata),
				embedding=[float(value) for value in embedding],
			)

	def get(self, ids=None, include=None, limit=None):
		if ids is None:
			selected_ids = list(self.records)
		else:
			selected_ids = [str(qa_id) for qa_id in ids if str(qa_id) in self.records]
		result = {
			"ids": selected_ids,
			"documents": [self.records[qa_id].document for qa_id in selected_ids],
			"metadatas": [self.records[qa_id].metadata for qa_id in selected_ids],
		}
		if include and "embeddings" in include:
			result["embeddings"] = [self.records[qa_id].embedding for qa_id in selected_ids]
		return result

	def update(self, *, ids, documents, metadatas, embeddings=None) -> None:
		for index, qa_id in enumerate(ids):
			record = self.records[str(qa_id)]
			record.document = documents[index]
			record.metadata = dict(metadatas[index])
			if embeddings is not None:
				record.embedding = [float(value) for value in embeddings[index]]

	def delete(self, *, ids) -> None:
		for qa_id in ids:
			self.records.pop(str(qa_id), None)

	def query(self, *, query_embeddings, n_results, include=None):
		query_embedding = [float(value) for value in query_embeddings[0]]
		scored_records = [
			(
				record,
				1.0 - _cosine_similarity(query_embedding, record.embedding),
			)
			for record in self.records.values()
		]
		scored_records.sort(key=lambda item: item[1])
		selected = scored_records[:n_results]
		return {
			"ids": [[record.id for record, _distance in selected]],
			"documents": [[record.document for record, _distance in selected]],
			"metadatas": [[record.metadata for record, _distance in selected]],
			"distances": [[distance for _record, distance in selected]],
		}


class _FakeClient:
	def __init__(self, *_args, **_kwargs) -> None:
		self.collection = _FakeCollection()

	def get_or_create_collection(self, *, name, metadata):
		return self.collection


def _fake_import(name: str):
	if name == "chromadb":
		return SimpleNamespace(PersistentClient=_FakeClient)
	if name == "chromadb.config":
		return SimpleNamespace(Settings=lambda **kwargs: kwargs)
	return __import__(name)


@dataclass
class _LegacyQARecord:
	id: int
	question: str
	answer: str
	question_embedding: list[float]
	created_at: datetime
	updated_at: datetime


def test_get_store_respects_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
	store_module.get_store.cache_clear()
	monkeypatch.setattr(settings, "vector_db_backend", "chromadb")
	monkeypatch.setattr(store_module.importlib, "import_module", _fake_import)

	backend = store_module.get_store()

	assert isinstance(backend, ChromaQAStore)
	store_module.get_store.cache_clear()


def test_chroma_store_crud_and_search(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(store_module.importlib, "import_module", _fake_import)
	monkeypatch.setattr(settings, "chroma_persist_directory", ".chroma-test")
	monkeypatch.setattr(settings, "chroma_collection_name", "qa_test_records")

	store = ChromaQAStore()
	store.initialize()

	created = store.create_qa(question="What is the policy?", answer="Policy answer", question_embedding=[1.0, 0.0])
	assert created.id == 1
	assert store.get_qa_by_id(1).answer == "Policy answer"

	updated = store.update_qa(qa_id=1, question="What is the updated policy?", answer="Updated answer", question_embedding=[0.0, 1.0])
	assert updated is not None
	assert updated.question == "What is the updated policy?"
	assert updated.answer == "Updated answer"

	results = store.search_similar_questions(query_embedding=[0.0, 1.0], min_score=0.85, top_k=5)
	assert len(results) == 1
	assert results[0][0].id == 1
	assert results[0][1] >= 0.85

	assert store.delete_qa(1) is True
	assert store.get_qa_by_id(1) is None


def test_chroma_store_syncs_legacy_postgres_records(monkeypatch: pytest.MonkeyPatch) -> None:
	legacy_record = _LegacyQARecord(
		id=7,
		question="provide vector db for simple search",
		answer="Use PostgreSQL with pgvector or ChromaDB.",
		question_embedding=[1.0, 0.0],
		created_at=datetime.now(timezone.utc),
		updated_at=datetime.now(timezone.utc),
	)

	class FakePostgresStore:
		def iter_full_records(self):
			return [legacy_record]

	monkeypatch.setattr(store_module.importlib, "import_module", _fake_import)
	monkeypatch.setattr(store_module, "PostgresQAStore", FakePostgresStore)
	monkeypatch.setattr(settings, "chroma_persist_directory", ".chroma-test")
	monkeypatch.setattr(settings, "chroma_collection_name", "qa_test_records")

	store = ChromaQAStore()
	store.initialize()

	results = store.search_similar_questions(query_embedding=[1.0, 0.0], min_score=0.85, top_k=5)

	assert len(results) == 1
	assert results[0][0].id == 7
	assert results[0][0].question == legacy_record.question
	assert results[0][0].answer == legacy_record.answer
	assert results[0][1] >= 0.85


def test_postgres_backend_is_still_available() -> None:
	assert PostgresQAStore is not None