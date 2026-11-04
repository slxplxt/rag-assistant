
import json
from datetime import datetime, timezone

from config import BASE_DIR, EMBEDDING_MODEL, GIGACHAT_MODEL
from query_data import init_rag_system
from rag_service import answer_question
from evaluation.validate_dataset import load_dataset


K = 3


def main():
    dataset_path = BASE_DIR / "evaluation" / "datasets" / "dev.json"
    output_path = BASE_DIR / "evaluation" / "results" / "generation_baseline.json"

    samples = load_dataset(dataset_path)
    vector_db, llm, prompt_template = init_rag_system()

    results = []

    for sample in samples:
        result = answer_question(
            question=sample.question,
            vector_db=vector_db,
            llm=llm,
            prompt_template=prompt_template,
            k=K,
        )

        results.append({
            "id": sample.id,
            "question": sample.question,
            "answerable": sample.answerable,
            "reference": sample.expected_answer,
            "response": result.answer,
            "status": result.status,
            "retrieved_contexts": [
                source.content for source in result.sources
            ],
            "retrieved_sources": [
                {
                    "source": source.source,
                    "page": source.page,
                }
                for source in result.sources
            ],
        })

        print(f"{sample.id} | status={result.status}")

    report = {
        "experiment": "generation_baseline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "generation_model": GIGACHAT_MODEL,
        "retrieval_k": K,
        "total_questions": len(samples),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
