import json
import fitz
from pathlib import Path

from schemas import EvalSample

from config import BASE_DIR, PDF_PATH

dataset_path = BASE_DIR / "evaluation" / "datasets" / "dev.json"


def load_dataset(path: Path) -> list[EvalSample]:
    with path.open(encoding="utf-8") as file:
        raw_data = json.load(file)

    if not isinstance(raw_data, list):
        raise ValueError("Dataset must be a JSON array")

    samples = [
        EvalSample.model_validate(item)
        for item in raw_data
    ]

    ids = [sample.id for sample in samples]

    if len(ids) != len(set(ids)):
        raise ValueError("Ids isnt unique")
    
    return samples



def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def validate_evidence_in_pdf(
    samples: list[EvalSample],
    pdf_path: Path
) -> list[str]:

    errors = []

    with fitz.open(str(pdf_path)) as pdf:
        page_texts = [
            normalize_text(page.get_text("text", sort=True))
            for page in pdf
        ]

    for sample in samples:

        if not sample.answerable:
            continue

        relevant_texts = []

        for page_num in sample.relevant_pages:

            if not 1 <= page_num <= len(page_texts):
                errors.append(
                    f"{sample.id}: страница {page_num} не существует"
                )
                continue

            relevant_texts.append(page_texts[page_num - 1])

        for evidence in sample.relevant_evidence:
            normalized_evidence = normalize_text(evidence)

            found = any(
                normalized_evidence in page_text
                for page_text in relevant_texts
            )

            if not found:
                errors.append(
                    f"{sample.id}: evidence не найден: {evidence}"
                )

    return errors


def main():
    samples = load_dataset(dataset_path)

    answerable_count = sum(
        sample.answerable for sample in samples
    )

    print(f"Total samples: {len(samples)}")
    print(f"Answerable: {answerable_count}")
    print(f"Unanswerable: {len(samples) - answerable_count}")

    errors = validate_evidence_in_pdf(
        samples=samples,
        pdf_path=Path(PDF_PATH)
    )

    if errors:
        print("\nDataset validation FAILED")

        for error in errors:
            print(f" - {error}")

        raise SystemExit(1)

    print("\nDataset validation PASSED")


if __name__ == "__main__":
    main()



