
from pathlib import Path

from schemas import RagResult, Source


class LLMProviderError(RuntimeError):
    """Ошибка при обращении к LLM-провайдеру."""
    pass


def detect_banks(question: str) -> set[str]:
    """Определяет банки, явно упомянутые в вопросе."""
    text = question.casefold()
    banks = set()

    if "втб" in text or "vtb" in text:
        banks.add("vtb")

    tbank_aliases = (
        "т-банк",
        "т-банка",
        "т-банке",
        "т банк",
        "т банка",
        "тинькофф",
        "tbank",
        "t-bank",
    )

    if any(alias in text for alias in tbank_aliases):
        banks.add("tbank")

    return banks


def answer_question(
    question: str,
    vector_db,
    llm,
    prompt_template,
    k: int = 3
) -> RagResult:

    if not question.strip():
        raise ValueError("Пустой вопрос")

    if k < 1:
        raise ValueError("k должно быть больше 0")

    banks = detect_banks(question)

    if "vtb" in banks:
        return RagResult(
            answer="MVP поддерживает только тариф Т-Банка ТПС 3.0. ВТБ и сравнение банков недоступны.",
            status="unsupported", sources=[],
        )
    search_options = {"filter": {"bank": "tbank"}} if banks else {}
    results = vector_db.similarity_search(question, k=k, **search_options)

    if not results:
        return RagResult(
            answer="В документах не найдена информация для ответа.",
            status="no_context",
            sources=[]
        )

    context_text = "\n\n".join(
        f"[Источник {i}, "
        f"банк: {doc.metadata.get('bank', 'unknown')}, "
        f"файл: {doc.metadata.get('source', 'unknown')}, "
        f"страница: {doc.metadata.get('page', '?')}]\n"
        f"{doc.page_content}"
        for i, doc in enumerate(results, start=1)
    )

    prompt = prompt_template.format(
        context=context_text,
        question=question
    )

    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        raise LLMProviderError(
            "Не удалось получить ответ от LLM"
        ) from exc

    sources = [
        Source(
            source=Path(
                str(doc.metadata.get("source", "unknown"))
            ).name,
            page=doc.metadata.get("page"),
            content=doc.page_content
        )
        for doc in results
    ]

    return RagResult(
        answer=response.content,
        status="answered",
        sources=sources
    )
