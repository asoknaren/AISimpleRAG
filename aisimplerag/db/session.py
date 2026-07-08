from collections.abc import Generator
from logging import getLogger

from aisimplerag.db.store import QAStore, get_store

logger = getLogger(__name__)

engine = None
SessionLocal = None


def init_db() -> None:
	store = get_store()
	logger.info("Initializing database backend: %s", store.__class__.__name__)
	store.initialize()


def get_db() -> Generator[QAStore, None, None]:
	yield get_store()
