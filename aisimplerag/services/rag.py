from sqlalchemy.orm import Session

from aisimplerag.api.schemas import RAGResponse, SearchMatch
from aisimplerag.core.config import settings
from aisimplerag.db.models.qa import QARecord
from aisimplerag.services.ollama import generate_with_ollama
from aisimplerag.services.openai import generate_with_openai
from aisimplerag.services.retrieval import retrieve_matches


def _fallback_answer(question: str, matches: list[tuple[QARecord, float]]) -> str:
    if not matches:
        return (
            "No relevant QA context met the configured similarity threshold. "
            "Try rephrasing the question."
        )
    top_match = matches[0][0]
    provider_hint = (
        "Verify OpenAI credentials and endpoint configuration in environment variables."
        if settings.use_openai
        else "Start or verify Ollama to receive a fuller synthesized explanation tailored to your exact wording."
    )
    return (
        f"Answer: Based on the closest stored QA, {top_match.answer}\n"
        "Additional context:\n"
        "- This response was generated from retrieval fallback because the LLM generator was unavailable.\n"
        f"- {provider_hint}"
    )


def _build_prompt(question: str, matches: list[tuple[QARecord, float]]) -> str:
    if not matches:
        return (
            "You are a QA assistant. "
            "Answer the user's question directly, then add 1-2 short points of helpful background context. "
            "Do not copy source text verbatim.\n\n"
            f"User question:\n{question}\n"
        )

    context_lines: list[str] = []
    for index, (record, similarity) in enumerate(matches, start=1):
        context_lines.append(
            f"[{index}] Similarity: {similarity:.4f}\n"
            f"Question: {record.question}\n"
            f"Answer: {record.answer}"
        )

    context_text = "\n\n".join(context_lines)
    return (
        "You are a concise QA assistant. Use the retrieved QA context below as grounding, but synthesize a fresh answer in your own words. "
        "Do not copy any retrieved answer sentence verbatim unless quoting is explicitly needed. "
        "After the direct answer, add a short 'Additional context' section with 1-2 useful facts relevant to the user's question. "
        "If context is insufficient, clearly say what is missing.\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        f"User question:\n{question}\n\n"

        #"Return exactly this format:\n"
        #"Answer: <direct answer>\n"
        #"Additional context:\n"
        #"- <fact 1>\n"
        #"- <fact 2 if useful>"
    )


def build_rag_response(db: Session, *, question: str) -> RAGResponse:
    matches = retrieve_matches(db, question=question)
    prompt = _build_prompt(question, matches)
    try:
        if settings.use_openai:
            generated_answer = generate_with_openai(prompt=prompt)
        else:
            generated_answer = generate_with_ollama(prompt=prompt)
    except Exception:
        # Keep the API responsive if the configured generator is unavailable.
        generated_answer = _fallback_answer(question, matches)

    return RAGResponse(
        question=question,
        generated_answer=generated_answer,
        matches=[
            SearchMatch(qa=record, similarity=similarity)
            for record, similarity in matches
        ],
    )
