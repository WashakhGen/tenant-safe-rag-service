import pytest

from app.benchmark import BENCHMARK
from app.evaluation import (
    answer_is_grounded,
    evaluate_benchmark,
    retrieval_recall_at_k,
)
from app.schemas import Evidence
from app.vector import vector_client

SOURCE = "Refunds are allowed within 7 calendar days of settlement."
REPORT_TIMES = "Settlement reports are generated daily at 02:00 UTC."


def _evidence(content=SOURCE, chunk_id="good"):
    return Evidence(chunk_id=chunk_id, account_id="atlas", content=content, score=0.9)


# answer_is_grounded
def test_verbatim_answer_with_required_fact_is_grounded():
    assert answer_is_grounded(SOURCE, [_evidence()], ["7 calendar days"]) is True


def test_short_answer_quoting_the_evidence_is_grounded():
    assert answer_is_grounded("7 calendar days", [_evidence()], ["7 calendar days"])


def test_case_and_whitespace_differences_are_ignored():
    answer = "REFUNDS  are allowed\nwithin 7   calendar days of settlement."
    assert answer_is_grounded(answer, [_evidence()], ["7 calendar days"]) is True


def test_extra_unsupported_claim_is_not_grounded():
    answer = SOURCE + " Refunds are always automatic."
    assert answer_is_grounded(answer, [_evidence()], ["7 calendar days"]) is False


def test_missing_required_fact_is_not_grounded():
    answer = "Refunds are allowed."
    assert answer_is_grounded(answer, [_evidence()], ["7 calendar days"]) is False


def test_fact_the_evidence_does_not_support_is_not_grounded():
    answer = "Refunds are allowed within 9 calendar days of settlement."
    assert answer_is_grounded(answer, [_evidence()], ["9 calendar days"]) is False


def test_fact_must_match_as_a_whole_phrase():
    # the answer is faithful to the evidence, but "17 calendar days" is not "7 calendar days"
    seventeen = _evidence("Refunds are allowed within 17 calendar days of settlement.")
    assert (
        answer_is_grounded(seventeen.content, [seventeen], ["7 calendar days"]) is False
    )


def test_contradicting_answer_is_not_grounded():
    answer = "Refunds are not allowed within 7 calendar days of settlement."
    assert answer_is_grounded(answer, [_evidence()], ["7 calendar days"]) is False


def test_no_evidence_or_empty_answer_is_not_grounded():
    assert answer_is_grounded(SOURCE, [], ["7 calendar days"]) is False
    assert answer_is_grounded("", [_evidence()], ["7 calendar days"]) is False


def test_abstention_is_not_a_grounded_answer():
    abstain = "I do not have enough evidence to answer that from the knowledge base."
    assert answer_is_grounded(abstain, [_evidence()], ["7 calendar days"]) is False


def test_multi_fact_answer_needs_every_fact_supported():
    evidence = [_evidence(SOURCE, "first"), _evidence(REPORT_TIMES, "second")]
    facts = ["7 calendar days", "02:00 UTC"]
    assert answer_is_grounded(f"{SOURCE} {REPORT_TIMES}", evidence, facts) is True
    assert answer_is_grounded(SOURCE, evidence, facts) is False
    assert answer_is_grounded(REPORT_TIMES, evidence, facts) is False


# grounding is separate from retrieval success


def test_good_retrieval_does_not_imply_a_grounded_answer():
    retrieved = [_evidence()]
    bad_answer = SOURCE + " Refunds are always automatic."
    assert retrieval_recall_at_k(retrieved, ["good"]) == 1.0
    assert answer_is_grounded(bad_answer, retrieved, ["7 calendar days"]) is False


def test_faithful_answer_from_the_wrong_chunk_is_not_grounded():
    wrong = _evidence(
        "Chargeback evidence is uploaded through the disputes console.", "wrong"
    )
    assert retrieval_recall_at_k([wrong], ["good"]) == 0.0
    assert answer_is_grounded(wrong.content, [wrong], ["7 calendar days"]) is False


def test_partial_recall_can_still_give_a_grounded_answer():
    # one of two relevant chunks retrieved, but the required fact lives in it
    retrieved = [_evidence(SOURCE, "first")]
    assert retrieval_recall_at_k(retrieved, ["first", "second"]) == 0.5
    assert answer_is_grounded(SOURCE, retrieved, ["7 calendar days"]) is True


# --- evaluate_benchmark ------------------------------------------------------


async def test_benchmark_baseline_is_perfect_and_leak_free():
    report = await evaluate_benchmark()
    assert report["n_cases"] == len(BENCHMARK)
    assert report["retrieval_recall_at_k"] == 1.0
    assert report["retrieval_mrr"] == 1.0
    assert report["grounded_answer_rate"] == 1.0
    assert report["tenant_leaks"] == 0
    assert all(case["grounded"] for case in report["cases"])


async def test_benchmark_is_deterministic():
    assert await evaluate_benchmark() == await evaluate_benchmark()


async def test_benchmark_top_k_limits_multi_chunk_questions():
    report = await evaluate_benchmark(top_k=1)
    b4 = next(case for case in report["cases"] if case["id"] == "b4")
    assert b4["retrieval_recall_at_k"] == 0.5  # only one of at1/at2 fits
    assert b4["grounded"] is False  # the other required fact is missing
    assert report["retrieval_recall_at_k"] < 1.0


async def test_benchmark_detects_retrieval_that_ignores_the_query(monkeypatch):
    async def storage_order(query, tenant_id, top_k):
        docs = vector_client.docs[tenant_id]
        return [Evidence.model_validate(d) for d in docs[:top_k]]

    monkeypatch.setattr(vector_client, "search", storage_order)
    report = await evaluate_benchmark(top_k=1)
    assert report["retrieval_recall_at_k"] < 1.0
    assert report["retrieval_mrr"] < 1.0
    assert report["grounded_answer_rate"] < 1.0


async def test_benchmark_reports_cross_tenant_leaks(monkeypatch):
    async def leaky(query, tenant_id, top_k):
        return [Evidence.model_validate(d) for d in vector_client.docs["cedar"]]

    monkeypatch.setattr(vector_client, "search", leaky)
    report = await evaluate_benchmark()
    assert report["tenant_leaks"] > 0


@pytest.mark.parametrize("case", BENCHMARK, ids=lambda c: c["id"])
async def test_each_benchmark_case_is_grounded(case):
    report = await evaluate_benchmark()
    row = next(r for r in report["cases"] if r["id"] == case["id"])
    assert row["grounded"] is True
