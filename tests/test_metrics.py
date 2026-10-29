import pytest 

from langchain_core.documents import Document

from schemas import EvalSample
from evaluation.metrics import (evidence_recall_at_k, all_evidence_success_at_k)

@pytest.fixture
def sample():
    return EvalSample(
        id="test_001",
        question="Сколько стоит оповещение об операциях?",
        answerable=True,
        expected_answer="59 или 99 рублей в месяц",
        relevant_pages=[1],
        relevant_evidence=[
            "5.1. с номером Т-Мобайл 59 руб. в месяц",
            "5.2. в прочих случаях 99 руб. в месяц",
        ],
        category="multi_condition",
    )


@pytest.fixture
def retrieved_docs():
    return [
        Document(
            page_content="5.1. с номером Т-Мобайл 59 руб. в месяц",
            metadata={"page": 1},
        ),
        Document(
            page_content="5.2. в прочих случаях 99 руб. в месяц",
            metadata={"page": 1},
        ),
    ]


def test_evidence_recall_at_k(sample, retrieved_docs):
    recall_at_1 = evidence_recall_at_k(
        sample=sample,
        retrieved_docs=retrieved_docs,
        k=1
    )

    recall_at_2 = evidence_recall_at_k(
        sample=sample,
        retrieved_docs=retrieved_docs,
        k=2,
    )

    assert recall_at_1 == 0.5
    assert recall_at_2 == 1

def test_all_evidence_succes(sample, retrieved_docs):
    success_at_1 = all_evidence_success_at_k(
        sample=sample,
        retrieved_docs=retrieved_docs,
        k=1
    )

    success_at_2 = all_evidence_success_at_k(
        sample=sample,
        retrieved_docs=retrieved_docs,
        k=2
    )

    assert success_at_1 == 0
    assert success_at_2 == 1

def test_no_evidence_found(sample):
    result = evidence_recall_at_k(
        sample=sample,
        retrieved_docs=[],
        k=2
    )

    success = all_evidence_success_at_k(
        sample=sample,
        retrieved_docs=[],
        k=2
    )

    assert result == 0
    assert success == 0



def test_unanswerable_question():
    sample = EvalSample(
        id="test_002",
        question="Сколько стоит доставка карты?",
        answerable=False,
        expected_answer=None,
        relevant_pages=[],
        relevant_evidence=[],
        category="unanswerable"
    )

    recall = evidence_recall_at_k(
        sample=sample,
        retrieved_docs=[],
        k=2
    )

    success = all_evidence_success_at_k(
        sample=sample,
        retrieved_docs=[],
        k=2
    )

    assert recall is None
    assert success is None


@pytest.mark.parametrize(
    "metric",
    [evidence_recall_at_k, all_evidence_success_at_k]
)
@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(sample, retrieved_docs, metric, k):
    with pytest.raises(ValueError):
        metric(
            sample=sample,
            retrieved_docs=retrieved_docs,
            k=k
        )


def test_wrong_page(sample):
    retrieved_docs = [
        Document(
            page_content="5.1. с номером Т-Мобайл 59 руб. в месяц",
            metadata={"page": 2},
        ),
        Document(
            page_content="5.2. в прочих случаях 99 руб. в месяц",
            metadata={"page": 2},
        ),
    ]

    recall = evidence_recall_at_k(
        sample=sample,
        retrieved_docs=retrieved_docs,
        k=2
    )

    assert recall == 0

def test_wrong_amount(sample):
    retrieved_docs = [
        Document(
            page_content="5.1. с номером Т-Мобайл 59 руб. в месяц",
            metadata={"page": 1},
        ),
        Document(
            page_content="5.2. в прочих случаях 199 руб. в месяц",
            metadata={"page": 1},
        ),
    ]

    recall = evidence_recall_at_k(
        sample=sample,
        retrieved_docs=retrieved_docs,
        k=2
    )

    success = all_evidence_success_at_k(
        sample=sample,
        retrieved_docs=retrieved_docs,
        k=2
    )

    assert recall == 0.5
    assert success == 0
