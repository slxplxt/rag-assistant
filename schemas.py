from pydantic import BaseModel, Field, model_validator
from typing import Literal

class EvidenceJudgment(BaseModel):
    """Результат проверки того, подтверждает ли фрагмент документа заданный факт."""

    is_supported: bool = Field(
        description="Подтверждает ли фрагмент документа проверяемый факт полностью"
    )
    reason: str = Field(
        description="Краткое объяснение решения"
    )


class Source(BaseModel):
    source: str
    page: int | None = None
    content: str


class RagResult(BaseModel):
    answer: str
    status: Literal["answered", "no_context"]
    sources: list[Source] = Field(default_factory=list)

class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Вопрос пользователя по банковскому тарифу"
    )

class EvalSample(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)

    answerable: bool
    expected_answer: str | None

    relevant_pages: list[int]
    relevant_evidence: list[str]

    category: str

    @model_validator(mode="after")
    def validate_evidence(self):
        if self.answerable:
            if not self.expected_answer:
                raise ValueError(
                    "Answerable question must have expected_answer"
                )

            if not self.relevant_evidence:
                raise ValueError(
                    "Answerable question must have relevant_evidence"
                )

            if not self.relevant_pages:
                raise ValueError(
                    "Answerable question must have relevant_pages"
                )
        else:
            if (
                self.expected_answer is not None
                or self.relevant_evidence
                or self.relevant_pages
            ):
                raise ValueError(
                    "Unanswerable question must not contain "
                    "an answer or evidence"
                )

        return self