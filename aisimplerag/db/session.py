from collections.abc import Generator
from logging import getLogger
from socket import create_connection

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from aisimplerag.core.config import settings
from aisimplerag.db.base import Base

# Ensure model metadata is registered before initialization.
from aisimplerag.db import models  # noqa: F401

logger = getLogger(__name__)

engine = create_engine(
	settings.database_url,
	future=True,
	pool_pre_ping=True,
	connect_args={"connect_timeout": 5},
)
SessionLocal = sessionmaker(
	bind=engine,
	autoflush=False,
	autocommit=False,
	expire_on_commit=False,
	class_=Session,
	future=True,
)


def init_db() -> None:
	logger.info("Initializing database")
	try:
		with create_connection((settings.postgres_host, settings.postgres_port), timeout=5):
			pass
	except OSError as exc:
		raise RuntimeError(
			f"Unable to reach PostgreSQL at {settings.postgres_host}:{settings.postgres_port}. "
			"Start PostgreSQL before launching the API."
		) from exc
	with engine.begin() as connection:
		connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
		if settings.postgres_schema != "public":
			connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{settings.postgres_schema}"'))

	Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()
