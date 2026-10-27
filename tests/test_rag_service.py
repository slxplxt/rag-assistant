from unittest.mock import Mock
import pytest
from langchain_core.documents import Document

from rag_service import answer_question

document = Document(
    page_content="Стоимость услуги — 99 рублей в месяц.",
    metadata={
        "source": "tariff.pdf",
        "page": 1
    }
)


def test_no_context_does_not_call_llm():
    vector_db = Mock()
    vector_db.similarity_search.return_value = []

    llm = Mock()
    prompt_template = Mock()

    result = answer_question(
        question="Сколько стоит обслуживание?",
        vector_db=vector_db,
        llm=llm,
        prompt_template=prompt_template,
        k=3
    )

    assert result.status == "no_context"
    assert result.sources == []

    vector_db.similarity_search.assert_called_once_with(
        "Сколько стоит обслуживание?",
        k=3
    )

    llm.invoke.assert_not_called()


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_raise_value_error(question):
    vector_db = Mock()
    llm = Mock()
    prompt_template = Mock()

    with pytest.raises(ValueError):
        answer_question(
            question=question,
            vector_db=vector_db,
            llm=llm,
            prompt_template=prompt_template,
            k=3
        )

    vector_db.similarity_search.assert_not_called()
    llm.invoke.assert_not_called()

def test_successful_generation():
    vector_db = Mock()
    vector_db.similarity_search.return_value = [document]

    llm = Mock()
    llm.invoke.return_value = Mock(
        content="Стоимость услуги составляет 99 рублей в месяц."
    )

    prompt_template = Mock()
    prompt_template.format.return_value = "Тестовый prompt"

    result = answer_question(
        question="Сколько стоит обслуживание?",
        vector_db=vector_db,
        llm=llm,
        prompt_template=prompt_template,
        k=3
    )

    assert result.status == "answered"
    assert result.answer == "Стоимость услуги составляет 99 рублей в месяц."

    assert len(result.sources) == 1
    assert result.sources[0].page == 1
    assert result.sources[0].source == "tariff.pdf"

    vector_db.similarity_search.assert_called_once_with(
        "Сколько стоит обслуживание?",
        k=3
    )

    llm.invoke.assert_called_once_with("Тестовый prompt")
    


