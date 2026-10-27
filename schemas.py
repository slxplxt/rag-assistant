from pydantic import BaseModel, Field
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