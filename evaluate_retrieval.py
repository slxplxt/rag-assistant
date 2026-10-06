import re
import json
from query_data import init_rag_system


STOP_WORDS = {
    "в", "во", "и", "или", "а",
    "на", "по", "с", "со",
    "к", "ко", "из", "для",
    "о", "об"
}


def normalize_text(text: str) -> str:
    text = text.lower().replace(",", ".")
    text = re.sub(
        r'(?<=\d)\s(?=\d{3}\b)',
        '',
        text
    )
    return " ".join(text.split())


def tokenize(text: str) -> set[str]:
    normalized = normalize_text(text)

    tokens = re.findall(
        r'\d+(?:\.\d+)?%?|[a-zа-яё]+',
        normalized
    )

    return {
        token
        for token in tokens
        if token not in STOP_WORDS
    }


def evidence_coverage(
    retrieved_text: str,
    relevant_text: str,
) -> float:
    if not relevant_text or not relevant_text.strip():
        return 0.0

    retrieved_tokens = tokenize(retrieved_text)
    relevant_tokens = tokenize(relevant_text)

    if not relevant_tokens:
        return 0.0

    intersection = retrieved_tokens & relevant_tokens

    return len(intersection) / len(relevant_tokens)

question = "Какой бесплатный лимит действует для переводов в другой банк по номеру карты через сервисы Т-Банка и какая комиссия предусмотрена в прочих случаях?"
relevant_evidence = [
      "7.4.1. до 20 000 руб. за расчетный период Бесплатно",
      "7.4.2. в прочих случаях 1,5%, минимум 30 руб."
    ]

def is_evidence_found(
        evidence: str,
        retrieved_docs,
        threshold: float
) -> bool:
    for doc in retrieved_docs:
        if evidence_coverage(doc.page_content, evidence) >= threshold:
            return True
    return False

def count_found_evidence(
    relevant_evidence: list[str],
    retrieved_docs,
    threshold: float,
) -> int:
    found = 0

    for evidence in relevant_evidence:
        if is_evidence_found(
            evidence,
            retrieved_docs,
            threshold
        ):
            found += 1

    return found

def evidence_recall_at_k(
    relevant_evidence: list[str],
    retrieved_docs,
    threshold: float,
) -> float | None:
    
    if not relevant_evidence:
        return None

    found = count_found_evidence(
        relevant_evidence, 
        retrieved_docs,
        threshold
    )

    return found / len(relevant_evidence)

def evaluate_mean_recall_at_k(
        eval_data,
        vector_db,
        k: int,
        threshold: float
) -> float:
    recalls = []
    for item in eval_data:
        results = vector_db.similarity_search(query=item["question"], k=k)
        recall = evidence_recall_at_k(item["relevant_evidence"], results, threshold)
        if recall is not None:
            recalls.append(recall)

    if not recalls:
        return None 
    
    return sum(recalls) / len(recalls)

vector_db, llm, prompt = init_rag_system()

with open("data/tbank_rag_eval.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

for k in [1, 3, 5]:
    recall = evaluate_mean_recall_at_k(
        eval_data=eval_data,
        vector_db=vector_db,
        k=k,
        threshold=0.8
    )
    print(f"Recall@{k}: {recall}")