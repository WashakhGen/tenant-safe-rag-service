import pytest

from app.evaluation import answer_is_grounded, retrieval_recall_at_k
from app.llm import model_client
from app.schemas import Evidence


def _evidence(
    chunk_id="good",
    content="Refunds are allowed within 7 calendar days of settlement.",
    score=0.9,
):
    return Evidence(chunk_id=chunk_id, account_id="atlas", content=content, score=score)


def test_retrieval_metric_does_not_count_irrelevant_evidence_as_relevant():
    assert (
        retrieval_recall_at_k([_evidence("wrong", "Unrelated content.")], ["good"])
        == 0.0
    )


def test_grounding_metric_rejects_unsupported_claim():
    source = "Refunds are allowed within 7 calendar days of settlement."
    answer = source + " Refunds are always automatic."
    assert (
        answer_is_grounded(answer, [_evidence("good", source)], ["7 calendar days"])
        is False
    )


@pytest.mark.asyncio
async def test_multi_fact_answer_uses_multiple_retrieved_evidence_items():
    first = _evidence(
        "first", "Refunds are allowed within 7 calendar days of settlement.", 0.9
    )
    second = _evidence(
        "second", "Settlement reports are generated daily at 02:00 UTC.", 0.8
    )
    answer = await model_client.answer(
        "Give me the refund window and settlement report time.", [first, second]
    )
    assert "7 calendar days" in answer
    assert "02:00 UTC" in answer
