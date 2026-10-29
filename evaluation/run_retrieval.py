
import json
from datetime import datetime, timezone
from pathlib import Path

from config import BASE_DIR, EMBEDDING_MODEL
from query_data import init_vector_db
from evaluation.validate_dataset import load_dataset
from evaluation.metrics import (
    evidence_recall_at_k,
    all_evidence_success_at_k,
)

K_VALUES = (1, 2, 3)


def main():
    dataset_path = BASE_DIR / "evaluation" / "datasets" / "dev.json"
    output_path = BASE_DIR / "evaluation" / "results" / "dense_baseline.json"

    samples = load_dataset(dataset_path)
    vector_db = init_vector_db()

    scores = {
        k: {"recall": [], "success": []}
        for k in K_VALUES
    }

    results = []

    for sample in samples:
        if not sample.answerable:
            continue

        docs = vector_db.similarity_search(
            sample.question,
            k=max(K_VALUES),
        )

        question_metrics = {}

        for k in K_VALUES:
            recall = evidence_recall_at_k(
                sample=sample,
                retrieved_docs=docs,
                k=k,
            )

            success = all_evidence_success_at_k(
                sample=sample,
                retrieved_docs=docs,
                k=k,
            )

            scores[k]["recall"].append(recall)
            scores[k]["success"].append(success)

            question_metrics[str(k)] = {
                "evidence_recall": recall,
                "all_evidence_success": success,
            }

        results.append({
            "id": sample.id,
            "question": sample.question,
            "metrics": question_metrics,
            "retrieved_chunks": [
                {
                    "rank": rank,
                    "page": doc.metadata.get("page"),
                    "source": Path(
                        str(doc.metadata.get("source", "unknown"))
                    ).name,
                    "content": doc.page_content,
                }
                for rank, doc in enumerate(docs, start=1)
            ],
        })

    if not results:
        raise ValueError("No answerable samples to evaluate")

    summary = {}

    for k in K_VALUES:
        mean_recall = sum(scores[k]["recall"]) / len(scores[k]["recall"])
        success_rate = sum(scores[k]["success"]) / len(scores[k]["success"])

        summary[str(k)] = {
            "mean_evidence_recall": round(mean_recall, 4),
            "all_evidence_success_rate": round(success_rate, 4),
        }

        print(
            f"k={k} | "
            f"Mean Recall={mean_recall:.3f} | "
            f"All-Evidence Success={success_rate:.3f}"
        )

    report = {
        "experiment": "dense_baseline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "evaluation/datasets/dev.json",
        "embedding_model": EMBEDDING_MODEL,
        "retrieval_method": "dense",
        "k_values": list(K_VALUES),
        "total_questions": len(samples),
        "evaluated_questions": len(results),
        "summary": summary,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    main()
