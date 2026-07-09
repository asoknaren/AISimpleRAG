"""Service layer package."""

from aisimplerag.services.ollama import generate_with_ollama
from aisimplerag.services.openai import generate_with_openai

__all__ = ["generate_with_ollama", "generate_with_openai"]
