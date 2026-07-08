from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL_NAME",
    )
    embedding_dimension: int = Field(default=384, validation_alias="EMBEDDING_DIMENSION")
    search_min_score: float = Field(default=0.45, validation_alias="SEARCH_MIN_SCORE")
    search_top_k: int = Field(default=5, validation_alias="SEARCH_TOP_K")

    vector_db_backend: str = Field(default="postgresql", validation_alias="VECTOR_DB_BACKEND")

    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_model_name: str = Field(default="llama3.1:8b", validation_alias="OLLAMA_MODEL_NAME")
    ollama_timeout_seconds: float = Field(default=30.0, validation_alias="OLLAMA_TIMEOUT_SECONDS")

    chroma_persist_directory: str = Field(default=".chroma", validation_alias="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="qa_records", validation_alias="CHROMA_COLLECTION_NAME")

    postgres_host: str = Field(default="127.0.0.1", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="aisimplerag", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", validation_alias="POSTGRES_PASSWORD")
    postgres_schema: str = Field(default="public", validation_alias="POSTGRES_SCHEMA")

    @property
    def database_url(self) -> str:
        return (
            "postgresql+psycopg://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
