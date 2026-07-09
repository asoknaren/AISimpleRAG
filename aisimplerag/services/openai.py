from collections.abc import Sequence

import httpx

from aisimplerag.core.config import settings


def _headers() -> dict[str, str]:
    if not settings.openai_api_key.strip():
        raise RuntimeError("OPENAI_API_KEY is required when AI_PROVIDER=openai.")
    return {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }


def _normalize_embedding(embedding: Sequence[float]) -> list[float]:
    return [float(value) for value in embedding]


def generate_openai_embedding(text: str) -> list[float]:
    payload = {
        "model": settings.openai_embedding_model,
        "input": text,
    }
    # Keep vector shape aligned with configured pgvector column size.
    if settings.openai_embedding_model.strip().startswith("text-embedding-3"):
        payload["dimensions"] = settings.embedding_dimension
    with httpx.Client(base_url=settings.openai_base_url.rstrip("/"), timeout=settings.openai_timeout_seconds) as client:
        response = client.post("/embeddings", json=payload, headers=_headers())
        response.raise_for_status()

    data = response.json()
    embedding = data.get("data", [{}])[0].get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError("OpenAI embeddings response did not include embedding data.")
    return _normalize_embedding(embedding)


def generate_with_openai(*, prompt: str) -> str:
    payload = {
        "model": settings.openai_generation_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise QA assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
        "max_tokens": settings.openai_max_tokens,
    }
    with httpx.Client(base_url=settings.openai_base_url.rstrip("/"), timeout=settings.openai_timeout_seconds) as client:
        response = client.post("/chat/completions", json=payload, headers=_headers())
        response.raise_for_status()

    data = response.json()
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenAI chat completion response did not include choices.")

    answer = choices[0].get("message", {}).get("content")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("OpenAI chat completion response did not include generated text.")
    return answer.strip()