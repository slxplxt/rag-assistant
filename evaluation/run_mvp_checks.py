"""Small reproducible smoke check; not a representative benchmark."""
import argparse
import hashlib
import json
from datetime import datetime, timezone

from config import BASE_DIR, PDF_PATH, EMBEDDING_MODEL, GIGACHAT_MODEL
from evaluation.metrics import evidence_recall_at_k
from evaluation.validate_dataset import load_dataset, validate_evidence_in_pdf
from query_data import init_vector_db, init_rag_system
from rag_service import answer_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Makes three GigaChat requests")
    args = parser.parse_args()
    samples = load_dataset(BASE_DIR / "evaluation/datasets/dev.json")
    errors = validate_evidence_in_pdf(samples, PDF_PATH)
    if errors:
        raise ValueError(errors)
    db = init_vector_db()
    rows = []
    for sample in samples:
        docs = db.similarity_search(sample.question, k=3)
        assert all(d.metadata.get("source") == PDF_PATH.name and d.metadata.get("bank") == "tbank" for d in docs)
        rows.append({"id": sample.id, "question": sample.question,
                     "recall": {str(k): evidence_recall_at_k(sample, docs, k) for k in (1, 2, 3)}})
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "pdf_sha256": hashlib.sha256(PDF_PATH.read_bytes()).hexdigest(),
              "embedding_model": EMBEDDING_MODEL, "generation_model": GIGACHAT_MODEL,
              "chunk_count": len(db.get()["ids"]), "retrieval": rows,
              "generation": [], "generation_error": None}
    if args.generate:
        try:
            db, llm, prompt = init_rag_system()
            for question in [
                "Сколько стоит обслуживание Т-Банка при невыполнении условий бесплатности?",
                "При каких условиях обслуживание карты Т-Банка бесплатно?",
                "Сколько дней занимает доставка карты Т-Банка?",
            ]:
                result = answer_question(question, db, llm, prompt)
                report["generation"].append({"question": question, **result.model_dump()})
        except Exception as exc:
            # Do not serialize provider errors: they may include credentials or request data.
            report["generation_error"] = type(exc).__name__
    path = BASE_DIR / "evaluation/results/mvp_smoke.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {path}; generation responses: {len(report['generation'])}; error: {report['generation_error']}")
    if report["generation_error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
