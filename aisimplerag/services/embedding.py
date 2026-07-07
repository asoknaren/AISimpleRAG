from collections.abc import Sequence

from sentence_transformers import SentenceTransformer

from aisimplerag.core.config import settings

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.embedding_model_name)
    return _embedding_model


def normalize_embedding(embedding: Sequence[float]) -> list[float]:
    return [float(value) for value in embedding]


def generate_question_embedding(question: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(question, normalize_embeddings=True)
    return normalize_embedding(embedding)

