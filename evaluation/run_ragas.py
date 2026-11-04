
import asyncio
import json
import math

from datetime import datetime, timezone
from statistics import mean

from config import BASE_DIR
from query_data import init_llm

from langchain_core.exceptions import OutputParserException

from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, FactualCorrectness


METRIC_NAMES = ("faithfulness", "factual_correctness")


def summarize_metric(results: list[dict], metric_name: str) -> dict:
    valid_scores = [
        row[metric_name]
        for row in results
        if row[metric_name] is not None
    ]

    return {
        "mean": mean(valid_scores) if valid_scores else None,
        "scored": len(valid_scores),
        "failed": len(results) - len(valid_scores),
    }


def save_report(
    output_path,
    results: list[dict],
    total_questions: int,
) -> None:

    output_report = {
        "experiment": "ragas_baseline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ragas_version": "0.4.3",
        "judge_model": "GigaChat",
        "total_questions": total_questions,
        "evaluated_questions": len(results),
        "summary": {
            metric_name: summarize_metric(results, metric_name)
            for metric_name in METRIC_NAMES
        },
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output_report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


async def main():
    report_path = (
        BASE_DIR
        / "evaluation"
        / "results"
        / "generation_baseline.json"
    )

    output_path = (
        BASE_DIR
        / "evaluation"
        / "results"
        / "ragas_baseline.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    answerable_rows = [
        row for row in report["results"]
        if row["answerable"]
    ]

    if not answerable_rows:
        raise ValueError("No answerable questions to evaluate")

    judge_llm = LangchainLLMWrapper(init_llm())

    metrics = [
        ("faithfulness", Faithfulness(llm=judge_llm)),
        (
            "factual_correctness",
            FactualCorrectness(llm=judge_llm, mode="f1"),
        ),
    ]

    results = []

    for row in answerable_rows:
        sample = SingleTurnSample(
            user_input=row["question"],
            response=row["response"],
            reference=row["reference"],
            retrieved_contexts=row["retrieved_contexts"],
        )

        question_result = {
            "id": row["id"],
            "faithfulness": None,
            "factual_correctness": None,
            "errors": {},
        }

        for metric_name, metric in metrics:
            try:
                score = await metric.single_turn_ascore(sample)
                score = float(score)

                if math.isfinite(score):
                    question_result[metric_name] = score
                else:
                    question_result["errors"][metric_name] = (
                        "Non-finite score"
                    )

            except OutputParserException:
                question_result["errors"][metric_name] = (
                    "OutputParserException"
                )

                print(
                    f"{row['id']} | "
                    f"{metric_name}: output parsing failed"
                )

        results.append(question_result)

        save_report(
            output_path=output_path,
            results=results,
            total_questions=len(answerable_rows),
        )

        faith_score = question_result["faithfulness"]
        correctness_score = question_result["factual_correctness"]

        print(
            f"{row['id']} | "
            f"Faithfulness="
            f"{f'{faith_score:.3f}' if faith_score is not None else 'FAILED'} | "
            f"Factual Correctness="
            f"{f'{correctness_score:.3f}' if correctness_score is not None else 'FAILED'}"
        )

    print("\nRagas Evaluation Summary")

    for metric_name in METRIC_NAMES:
        summary = summarize_metric(results, metric_name)

        mean_score = summary["mean"]
        mean_text = (
            f"{mean_score:.3f}"
            if mean_score is not None
            else "N/A"
        )

        print(
            f"{metric_name}: "
            f"mean={mean_text} | "
            f"scored={summary['scored']} | "
            f"failed={summary['failed']}"
        )

    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
