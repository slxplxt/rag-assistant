from pathlib import Path

from schemas import RagResult, Source

def answer_question(question: str, vector_db, llm, prompt_template, k: int=3) -> RagResult:
    if not question.strip():
        raise ValueError("Пустой вопрос")

    if k < 1:
        raise ValueError("k должно быть больше 0")
    
    results = vector_db.similarity_search(question, k=k)

    if not results:
        return RagResult(
            answer="В документах не найдена информация для ответа.",
            status="no_context",
            sources=[]
        )

    context_text = "\n\n".join(
                f"[Источник {i}, страница {doc.metadata.get('page', '?')}]\n"
                f"{doc.page_content}"
                for i, doc in enumerate(results, start=1)
                )

    prompt = prompt_template.format(context=context_text, question=question)
    response = llm.invoke(prompt)

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
    return RagResult(answer=response.content, status="answered", sources=sources)