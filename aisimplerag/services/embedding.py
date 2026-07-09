from collections.abc import Sequence

from aisimplerag.core.config import settings
from aisimplerag.services.openai import generate_openai_embedding

_embedding_model = None


def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        sentence_transformer = _load_sentence_transformer()
        _embedding_model = sentence_transformer(settings.embedding_model_name)
    return _embedding_model


def normalize_embedding(embedding: Sequence[float]) -> list[float]:
    return [float(value) for value in embedding]


def _ensure_expected_dimension(embedding: Sequence[float]) -> None:
    expected = settings.embedding_dimension
    actual = len(embedding)
    if actual != expected:
        provider = "openai" if settings.use_openai else "local"
        raise ValueError(
            f"Embedding dimension mismatch for provider '{provider}': expected {expected}, got {actual}. "
            "Set EMBEDDING_DIMENSION to match your model output, or choose a model/configuration that returns the configured size."
        )


def generate_question_embedding(question: str) -> list[float]:
    if settings.use_openai:
        embedding = generate_openai_embedding(question)
        _ensure_expected_dimension(embedding)
        return embedding

    model = get_embedding_model()
    embedding = model.encode(question, normalize_embeddings=True)
    normalized = normalize_embedding(embedding)
    _ensure_expected_dimension(normalized)
    return normalized
