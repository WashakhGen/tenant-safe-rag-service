import pytest

from app.benchmark import BENCHMARK
from app.vector import vector_client


def _ids(results):
    return [r.chunk_id for r in results]


async def test_query_terms_decide_the_order_not_storage_order():
    # at1 (refund) is stored first, but the question is about settlement
    results = await vector_client.search("settlement files", "atlas", 3)
    assert _ids(results)[0] == "at2"


async def test_top_k_is_applied_after_ranking():
    results = await vector_client.search("settlement", "atlas", 1)
    assert _ids(results) == ["at2"]


async def test_plural_and_singular_terms_match():
    results = await vector_client.search("refunds", "atlas", 3)
    assert _ids(results) == ["at1"]


async def test_filler_words_alone_match_nothing():
    assert await vector_client.search("what is the", "atlas", 3) == []


async def test_unrelated_query_returns_nothing():
    assert await vector_client.search("weather in paris", "atlas", 3) == []


async def test_more_matching_terms_rank_higher():
    results = await vector_client.search("refund settlement files", "atlas", 3)
    assert _ids(results) == ["at2", "at1"]
    assert results[0].score > results[1].score


async def test_ranking_is_deterministic():
    first = await vector_client.search("refund settlement", "atlas", 3)
    second = await vector_client.search("refund settlement", "atlas", 3)
    assert first == second


async def test_ranking_never_crosses_tenants():
    results = await vector_client.search("refund settlement", "cedar", 5)
    assert _ids(results) == ["cd1", "cd2"]
    assert {r.account_id for r in results} == {"cedar"}


@pytest.mark.parametrize("case", BENCHMARK, ids=lambda c: c["id"])
async def test_benchmark_questions_retrieve_their_relevant_chunks(case):
    results = await vector_client.search(case["question"], case["tenant"], 3)
    assert set(case["relevant_chunk_ids"]) <= set(_ids(results))
