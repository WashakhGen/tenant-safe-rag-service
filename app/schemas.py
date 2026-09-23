from typing import Any

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    account_override: str | None = None
    top_k: int = Field(default=3, ge=1, le=5)


class Evidence(BaseModel):
    chunk_id: str
    account_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    account_id: str
    answer: str
    sources: list[Evidence]
    degraded: bool = False
