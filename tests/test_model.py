from app.llm import NO_EVIDENCE_ANSWER, model_client
from app.schemas import Evidence


def _evidence(chunk_id, content):
    return Evidence(chunk_id=chunk_id, account_id="atlas", content=content, score=0.9)


async def test_answer_uses_every_retrieved_chunk():
    refund = _evidence("a", "Refunds are allowed within 7 calendar days of settlement.")
    report = _evidence("b", "Settlement reports are generated daily at 02:00 UTC.")
    answer = await model_client.answer(
        "refund window and report time?", [refund, report]
    )
    assert "7 calendar days" in answer
    assert "02:00 UTC" in answer


async def test_answer_keeps_the_ranked_order():
    first = _evidence("a", "First fact.")
    second = _evidence("b", "Second fact.")
    answer = await model_client.answer("q", [first, second])
    assert answer.index("First fact.") < answer.index("Second fact.")


async def test_duplicate_chunks_are_quoted_once():
    chunk = _evidence("a", "Only once.")
    answer = await model_client.answer("q", [chunk, chunk])
    assert answer.count("Only once.") == 1


async def test_no_evidence_gives_the_abstention_answer():
    assert await model_client.answer("q", []) == NO_EVIDENCE_ANSWER


async def test_blank_evidence_gives_the_abstention_answer():
    assert await model_client.answer("q", [_evidence("a", "   ")]) == NO_EVIDENCE_ANSWER
