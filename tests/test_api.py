from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from main import app
from rag_service import LLMProviderError
import pytest

def test_health():
    with patch(
        "main.init_rag_system",
        return_value=(Mock(), Mock(), Mock())
    ):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ask_success():
    vector_db = Mock()
    llm = Mock()
    prompt_template = Mock()

    expected_result = {
        "answer": "Стоимость услуги составляет 99 рублей.",
        "status": "answered",
        "sources": [
            {
                "source": "tariff.pdf",
                "page": 1,
                "content": "Стоимость услуги — 99 рублей."
            }
        ]
    }

    with patch(
        "main.init_rag_system",
        return_value=(vector_db, llm, prompt_template)
    ):
        with patch(
            "main.answer_question",
            return_value=expected_result
        ) as mock_answer:

            with TestClient(app) as client:
                response = client.post(
                    "/ask",
                    json={"question": "Сколько стоит обслуживание?"}
                )

            assert response.status_code == 200
            assert response.json() == expected_result

            mock_answer.assert_called_once_with(
                question="Сколько стоит обслуживание?",
                vector_db=vector_db,
                llm=llm,
                prompt_template=prompt_template,
                k=3
            )

def test_ask_invalid_question():
    vector_db = Mock()
    llm = Mock()
    prompt_template = Mock()

    with patch(
        "main.init_rag_system",
        return_value=(vector_db, llm, prompt_template)
    ):
        with patch("main.answer_question") as mock_answer:
            with TestClient(app) as client:
                response = client.post(
                    "/ask",
                    json={"question": ""}
                )

    assert response.status_code == 422
    mock_answer.assert_not_called()

def test_ask_llm_provider_error():
    vector_db = Mock()
    llm = Mock()
    prompt_template = Mock()

    with patch(
        "main.init_rag_system",
        return_value=(vector_db, llm, prompt_template)
    ):
        with patch(
            "main.answer_question",
            side_effect=LLMProviderError("Connection timeout")
        ) as mock_answer:
            with TestClient(app) as client:
                response = client.post(
                    "/ask",
                    json={"question": "Сколько стоит обслуживание?"}
                )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Сервис генерации временно недоступен"
    }

    assert "Connection timeout" not in response.text
    mock_answer.assert_called_once()


@pytest.mark.parametrize("question", [" ", "\n\t", "x" * 1001])
def test_invalid_input_never_calls_service(question):
    with patch("main.init_rag_system", return_value=(Mock(), Mock(), Mock())):
        with patch("main.answer_question") as answer:
            with TestClient(app) as client:
                assert client.post("/ask", json={"question": question}).status_code == 422
            answer.assert_not_called()


def test_unexpected_error_is_not_exposed():
    with patch("main.init_rag_system", return_value=(Mock(), Mock(), Mock())):
        with patch("main.answer_question", side_effect=RuntimeError("SECRET_INTERNAL_DETAIL")):
            with TestClient(app) as client:
                response = client.post("/ask", json={"question": "Тариф?"})
    assert response.status_code == 500
    assert "SECRET_INTERNAL_DETAIL" not in response.text
