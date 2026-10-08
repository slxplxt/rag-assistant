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


def best_evidence_match(evidence: str, retrieved_docs) -> tuple[float, int | None]:
    best_coverage = 0
    best_rank = None

    for rank, doc in enumerate(retrieved_docs, start=1):
        coverage = evidence_coverage(doc.page_content, evidence)

        if coverage > best_coverage:
            best_coverage = coverage
            best_rank = rank

    return (best_coverage, best_rank)

def first_relevant_rank(evidence, retrieved_docs, threshold) -> int | None:
    for rank, doc in enumerate(retrieved_docs, start=1):
        coverage = evidence_coverage(doc.page_content, evidence)
        if coverage >= threshold: 
            return rank
    return None

def first_full_coverage_rank(relevant_evidence: list[str], retrieved_docs, threshold: float) -> int | None:
    if not relevant_evidence:
        return None

    for rank in range(1, len(retrieved_docs) + 1):
        current_docs = retrieved_docs[:rank]
        if all(
            is_evidence_found(evidence, current_docs, threshold)
            for evidence in relevant_evidence
        ):
            return rank

    return None


vector_db, llm, prompt = init_rag_system()

with open("data/tbank_rag_eval.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

item = eval_data[6]

results = vector_db.similarity_search(query=item["question"], k=5)
rank = first_full_coverage_rank(item["relevant_evidence"], retrieved_docs=results, threshold=0.8)
print("First full coverage rank:", rank)

for evidence in item["relevant_evidence"]:
    rank = first_relevant_rank(
        evidence,
        results,
        threshold=0.8
    )
    print(f"Evidence: {evidence}")
    print(f"First relevant rank: {rank}")

for rank, doc in enumerate(results, start=1):
    print(f"\nRank {rank}")

    for evidence in item["relevant_evidence"]:
        coverage = evidence_coverage(doc.page_content, evidence)
        print(f"Coverage: {coverage:.3f}")
        print(f"Evidence: {evidence}")

evidence = item["relevant_evidence"][1]

for rank, doc in enumerate(results[:3], start=1):
    relevant_tokens = tokenize(evidence)
    retrieved_tokens = tokenize(doc.page_content)

    matched = relevant_tokens & retrieved_tokens
    missing = relevant_tokens - retrieved_tokens

    print(f"\nRank {rank}")
    print("Matched:", sorted(matched))
    print("Missing:", sorted(missing))

'''for k in [1, 3, 5]:
    recall = evaluate_mean_recall_at_k(
        eval_data=eval_data,
        vector_db=vector_db,
        k=k,
        threshold=0.8
    )
    print(f"Recall@{k}: {recall}")
'''