from fastapi import FastAPI
from logging import getLogger

from aisimplerag.api.router import api_router
from aisimplerag.core.logging import configure_logging
from aisimplerag.db.session import init_db

logger = getLogger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="AISimpleRAG")

    @app.on_event("startup")
    def startup() -> None:
        logger.info("Starting application startup")
        init_db()
        logger.info("Application startup complete")

    app.include_router(api_router)
    return app


app = create_app()
