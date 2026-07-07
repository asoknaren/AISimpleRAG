import httpx

from aisimplerag.core.config import settings


def generate_with_ollama(*, prompt: str) -> str:
    payload = {
        "model": settings.ollama_model_name,
        "prompt": prompt,
        "stream": False,
    }
    with httpx.Client(base_url=settings.ollama_base_url, timeout=settings.ollama_timeout_seconds) as client:
        response = client.post("/api/generate", json=payload)
        response.raise_for_status()

    data = response.json()
    answer = data.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Ollama response did not include generated text.")
    return answer.strip()
