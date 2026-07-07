from sqlalchemy.orm import Session

from aisimplerag.api.schemas import RAGResponse, SearchMatch
from aisimplerag.db.models.qa import QARecord
from aisimplerag.services.ollama import generate_with_ollama
from aisimplerag.services.retrieval import retrieve_matches


def _fallback_answer(question: str, matches: list[tuple[QARecord, float]]) -> str:
    if not matches:
        return (
            "No relevant QA context met the configured similarity threshold. "
            "Try rephrasing the question."
        )
    return matches[0][0].answer


def _build_prompt(question: str, matches: list[tuple[QARecord, float]]) -> str:
    if not matches:
        return (
            "You are a concise QA assistant. "
            "No supporting context was retrieved with the configured threshold. "
            "Answer the user honestly and suggest rephrasing if uncertain.\n\n"
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
        "You are a concise QA assistant. Use only the retrieved QA context below. "
        "If context is insufficient, clearly say so.\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        f"User question:\n{question}\n\n"
        "Final answer:"
    )


def build_rag_response(db: Session, *, question: str) -> RAGResponse:
    matches = retrieve_matches(db, question=question)
    prompt = _build_prompt(question, matches)
    try:
        generated_answer = generate_with_ollama(prompt=prompt)
    except Exception:
        # Keep the API responsive if Ollama is unavailable.
        generated_answer = _fallback_answer(question, matches)

    return RAGResponse(
        question=question,
        generated_answer=generated_answer,
        matches=[
            SearchMatch(qa=record, similarity=similarity)
            for record, similarity in matches
        ],
    )
