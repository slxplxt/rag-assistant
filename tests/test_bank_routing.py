
import pytest

from rag_service import detect_banks


@pytest.mark.parametrize(
    "question, expected",
    [
        (
            "Сколько стоит обслуживание Т-Банка?",
            {"tbank"},
        ),
        (
            "Какие тарифы у ВТБ?",
            {"vtb"},
        ),
        (
            "Сравни ВТБ и Т-Банк",
            {"vtb", "tbank"},
        ),
        (
            "Сколько стоит обслуживание карты?",
            set(),
        ),
    ],
)
def test_detect_banks(question, expected):
    assert detect_banks(question) == expected
