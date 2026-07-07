from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Protocol, runtime_checkable

from sqlalchemy import Select, create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from aisimplerag.core.config import settings
from aisimplerag.db.base import Base
from aisimplerag.db.models.qa import QARecord


@dataclass
class StoredQARecord:
	id: int
	question: str
	answer: str
	created_at: datetime
	updated_at: datetime


@runtime_checkable
class QAStore(Protocol):
	def initialize(self) -> None: ...

	def create_qa(
		self,
		*,
		question: str,
		answer: str,
		question_embedding: Sequence[float],
	) -> StoredQARecord: ...

	def get_qa_by_id(self, qa_id: int) -> StoredQARecord | None: ...

	def list_qa(self, *, limit: int = 100, offset: int = 0) -> list[StoredQARecord]: ...

	def update_qa(
		self,
		*,
		qa_id: int,
		question: str | None = None,
		answer: str | None = None,
		question_embedding: Sequence[float] | None = None,
	) -> StoredQARecord | None: ...

	def delete_qa(self, qa_id: int) -> bool: ...

	def search_similar_questions(
		self,
		*,
		query_embedding: Sequence[float],
		min_score: float | None = None,
		top_k: int | None = None,
	) -> list[tuple[StoredQARecord, float]]: ...


def _utc_now() -> datetime:
	return datetime.now(timezone.utc)


def _record_from_orm(record: QARecord) -> StoredQARecord:
	return StoredQARecord(
		id=record.id,
		question=record.question,
		answer=record.answer,
		created_at=record.created_at,
		updated_at=record.updated_at,
	)


class PostgresQAStore:
	def __init__(self) -> None:
		self.engine = create_engine(
			settings.database_url,
			future=True,
			pool_pre_ping=True,
			connect_args={"connect_timeout": 5},
		)
		self.session_factory = sessionmaker(
			bind=self.engine,
			autoflush=False,
			autocommit=False,
			expire_on_commit=False,
			class_=Session,
			future=True,
		)

	def initialize(self) -> None:
		with self.engine.begin() as connection:
			connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
			if settings.postgres_schema != "public":
				connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{settings.postgres_schema}"'))

		Base.metadata.create_all(bind=self.engine)

	def create_qa(
		self,
		*,
		question: str,
		answer: str,
		question_embedding: Sequence[float],
	) -> StoredQARecord:
		record = QARecord(
			question=question,
			answer=answer,
			question_embedding=[float(value) for value in question_embedding],
		)
		with self.session_factory() as db:
			db.add(record)
			db.commit()
			db.refresh(record)
			return _record_from_orm(record)

	def get_qa_by_id(self, qa_id: int) -> StoredQARecord | None:
		with self.session_factory() as db:
			record = db.get(QARecord, qa_id)
			return None if record is None else _record_from_orm(record)

	def list_qa(self, *, limit: int = 100, offset: int = 0) -> list[StoredQARecord]:
		stmt = (
			select(QARecord)
			.order_by(QARecord.id.asc())
			.offset(max(offset, 0))
			.limit(max(limit, 0))
		)
		with self.session_factory() as db:
			return [_record_from_orm(record) for record in db.scalars(stmt).all()]

	def update_qa(
		self,
		*,
		qa_id: int,
		question: str | None = None,
		answer: str | None = None,
		question_embedding: Sequence[float] | None = None,
	) -> StoredQARecord | None:
		with self.session_factory() as db:
			record = db.get(QARecord, qa_id)
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
			return _record_from_orm(record)

	def delete_qa(self, qa_id: int) -> bool:
		with self.session_factory() as db:
			record = db.get(QARecord, qa_id)
			if record is None:
				return False

			db.delete(record)
			db.commit()
			return True

	def search_similar_questions(
		self,
		*,
		query_embedding: Sequence[float],
		min_score: float | None = None,
		top_k: int | None = None,
	) -> list[tuple[StoredQARecord, float]]:
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

		with self.session_factory() as db:
			rows = db.execute(stmt).all()
			return [(_record_from_orm(record), float(similarity)) for record, similarity in rows]

	def iter_full_records(self) -> list[QARecord]:
		stmt = select(QARecord).order_by(QARecord.id.asc())
		with self.session_factory() as db:
			return list(db.scalars(stmt).all())


class ChromaQAStore:
	def __init__(self) -> None:
		self._client = None
		self._collection_instance = None
		self._next_id = 1

	def _import_chromadb(self):
		return importlib.import_module("chromadb")

	def _get_collection(self):
		if self._collection_instance is None:
			chromadb = self._import_chromadb()
			persist_directory = settings.chroma_persist_directory
			client = None
			if hasattr(chromadb, "PersistentClient"):
				client = chromadb.PersistentClient(path=persist_directory)
			else:
				config_module = importlib.import_module("chromadb.config")
				client = chromadb.Client(
					config_module.Settings(
						persist_directory=persist_directory,
						is_persistent=True,
					)
				)
			self._client = client
			self._collection_instance = client.get_or_create_collection(
				name=settings.chroma_collection_name,
				metadata={"hnsw:space": "cosine"},
			)
			self._sync_from_postgres_if_needed()
			self._next_id = self._derive_next_id()
		return self._collection_instance

	def _sync_from_postgres_if_needed(self) -> None:
		collection = self._collection_instance
		if collection is None:
			return

		current_count = getattr(collection, "count", lambda: 0)()
		if current_count:
			return

		postgres_store = PostgresQAStore()
		try:
			legacy_records = postgres_store.iter_full_records()
		except Exception:
			return

		if not legacy_records:
			return

		collection.add(
			ids=[str(record.id) for record in legacy_records],
			documents=[record.question for record in legacy_records],
			metadatas=[
				{
					"answer": record.answer,
					"created_at": record.created_at.isoformat(),
					"updated_at": record.updated_at.isoformat(),
				}
				for record in legacy_records
			],
			embeddings=[
				[float(value) for value in record.question_embedding]
				for record in legacy_records
			],
		)

	def _derive_next_id(self) -> int:
		collection = self._collection_instance
		if collection is None:
			return 1
		all_records = collection.get(include=["metadatas"], limit=100000)
		ids = [int(value) for value in all_records.get("ids", []) if str(value).isdigit()]
		return (max(ids) + 1) if ids else 1

	@staticmethod
	def _build_record(
		qa_id: int,
		question: str,
		answer: str,
		created_at: str | datetime,
		updated_at: str | datetime,
	) -> StoredQARecord:
		if isinstance(created_at, str):
			created_at = datetime.fromisoformat(created_at)
		if isinstance(updated_at, str):
			updated_at = datetime.fromisoformat(updated_at)
		return StoredQARecord(
			id=qa_id,
			question=question,
			answer=answer,
			created_at=created_at,
			updated_at=updated_at,
		)

	def initialize(self) -> None:
		self._get_collection()

	def create_qa(
		self,
		*,
		question: str,
		answer: str,
		question_embedding: Sequence[float],
	) -> StoredQARecord:
		collection = self._get_collection()
		now = _utc_now().isoformat()
		qa_id = self._next_id
		self._next_id += 1
		collection.add(
			ids=[str(qa_id)],
			documents=[question],
			metadatas=[{"answer": answer, "created_at": now, "updated_at": now}],
			embeddings=[[float(value) for value in question_embedding]],
		)
		return self._build_record(qa_id, question, answer, now, now)

	def get_qa_by_id(self, qa_id: int) -> StoredQARecord | None:
		collection = self._get_collection()
		result = collection.get(ids=[str(qa_id)], include=["documents", "metadatas"])
		if not result.get("ids"):
			return None
		return self._build_record(
			qa_id,
			result["documents"][0],
			result["metadatas"][0]["answer"],
			result["metadatas"][0]["created_at"],
			result["metadatas"][0]["updated_at"],
		)

	def list_qa(self, *, limit: int = 100, offset: int = 0) -> list[StoredQARecord]:
		collection = self._get_collection()
		result = collection.get(include=["documents", "metadatas"])
		records = [
			self._build_record(
				int(qa_id),
				document,
				metadata["answer"],
				metadata["created_at"],
				metadata["updated_at"],
			)
			for qa_id, document, metadata in zip(
				result.get("ids", []),
				result.get("documents", []),
				result.get("metadatas", []),
				strict=False,
			)
		]
		records.sort(key=lambda record: record.id)
		return records[max(offset, 0) : max(offset, 0) + max(limit, 0)]

	def update_qa(
		self,
		*,
		qa_id: int,
		question: str | None = None,
		answer: str | None = None,
		question_embedding: Sequence[float] | None = None,
	) -> StoredQARecord | None:
		record = self.get_qa_by_id(qa_id)
		if record is None:
			return None
		if question is not None and question_embedding is None:
			raise ValueError("question_embedding is required when question changes")

		updated_question = question if question is not None else record.question
		updated_answer = answer if answer is not None else record.answer
		updated_embedding = question_embedding

		now = _utc_now().isoformat()
		collection = self._get_collection()
		update_kwargs: dict[str, object] = {
			"ids": [str(qa_id)],
			"documents": [updated_question],
			"metadatas": [
				{"answer": updated_answer, "created_at": record.created_at.isoformat(), "updated_at": now}
			],
		}
		if updated_embedding is not None:
			update_kwargs["embeddings"] = [[float(value) for value in updated_embedding]]
		collection.update(**update_kwargs)
		return self._build_record(qa_id, updated_question, updated_answer, record.created_at, now)

	def delete_qa(self, qa_id: int) -> bool:
		collection = self._get_collection()
		record = self.get_qa_by_id(qa_id)
		if record is None:
			return False
		collection.delete(ids=[str(qa_id)])
		return True

	def search_similar_questions(
		self,
		*,
		query_embedding: Sequence[float],
		min_score: float | None = None,
		top_k: int | None = None,
	) -> list[tuple[StoredQARecord, float]]:
		score_threshold = settings.search_min_score if min_score is None else min_score
		max_results = settings.search_top_k if top_k is None else top_k
		collection = self._get_collection()
		result = collection.query(
			query_embeddings=[[float(value) for value in query_embedding]],
			n_results=max_results,
			include=["documents", "metadatas", "distances"],
		)
		if not result.get("ids"):
			return []

		matches: list[tuple[StoredQARecord, float]] = []
		for qa_id, document, metadata, distance in zip(
			result["ids"][0],
			result["documents"][0],
			result["metadatas"][0],
			result["distances"][0],
			strict=False,
		):
			similarity = 1.0 - float(distance)
			if similarity < score_threshold:
				continue
			matches.append(
				(
					self._build_record(
						int(qa_id),
						document,
						metadata["answer"],
						metadata["created_at"],
						metadata["updated_at"],
					),
					similarity,
				)
			)
		return matches[:max_results]


@lru_cache(maxsize=1)
def get_store() -> QAStore:
	backend = settings.vector_db_backend.strip().lower()
	if backend == "postgresql":
		return PostgresQAStore()
	if backend == "chromadb":
		return ChromaQAStore()
	raise ValueError(
		f"Unsupported VECTOR_DB_BACKEND '{settings.vector_db_backend}'. Use 'postgresql' or 'chromadb'."
	)