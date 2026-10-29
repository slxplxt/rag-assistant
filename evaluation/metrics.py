from langchain_core.documents import Document

from evaluation.validate_dataset import normalize_text
from schemas import EvalSample

def is_evidence_found(
        evidence: str,
        retrieved_docs: list[Document],
        relevent_pages: list[str]
) -> bool:
    normalized_evidence = normalize_text(evidence)

    return any(
        doc.metadata.get("page") in relevent_pages
        and normalized_evidence in normalize_text(doc.page_content)
        for doc in retrieved_docs
    )

def evidence_recall_at_k(
        sample: EvalSample,
        retrieved_docs: list[Document],
        k: int
) -> float | None:
    if k < 1:
        raise ValueError("K must be greater than 0")
    
    if not sample.answerable:
        return None

    top_k_docs = retrieved_docs[:k]

    found_count = sum(
        is_evidence_found(
            evidence=evidence,
            retrieved_docs=top_k_docs,
            relevent_pages=sample.relevant_pages
        )
        for evidence in sample.relevant_evidence
    )

    return found_count / len(sample.relevant_evidence)

def all_evidence_success_at_k(
    sample: EvalSample,
    retrieved_docs: list[Document],
    k: int
) -> int | None:

    recall = evidence_recall_at_k(
        sample=sample,
        retrieved_docs=retrieved_docs,
        k=k
    )

    if recall is None:
        return None

    return int(recall == 1.0)