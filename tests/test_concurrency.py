import asyncio
import time

import httpx
import pytest

from app.main import app
from app.vector import vector_client

DELAY = 0.1
REQUESTS = 10


@pytest.fixture
async def aclient(monkeypatch):
    monkeypatch.setattr(vector_client, "delay_seconds", DELAY)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _ask(client, tenant, question="What is the refund policy?"):
    return await client.post(
        "/v1/answer",
        headers={"X-Account-ID": tenant},
        json={"question": question},
    )


async def test_requests_do_not_block_each_other(aclient):
    started = time.perf_counter()
    responses = await asyncio.gather(*(_ask(aclient, "atlas") for _ in range(REQUESTS)))
    elapsed = time.perf_counter() - started

    assert all(r.status_code == 200 for r in responses)
    # Blocking sleeps would need REQUESTS * DELAY (1.0s); concurrent needs about DELAY.
    assert elapsed < REQUESTS * DELAY / 2


async def test_tenants_stay_isolated_under_concurrent_load(aclient):
    tenants = ["atlas", "cedar"] * 10
    responses = await asyncio.gather(*(_ask(aclient, t) for t in tenants))

    for tenant, r in zip(tenants, responses, strict=True):
        body = r.json()
        assert body["account_id"] == tenant
        assert body["sources"]
        assert all(s["account_id"] == tenant for s in body["sources"])
        expected = "5 calendar days" if tenant == "atlas" else "10 calendar days"
        assert expected in body["answer"]


async def test_concurrent_identical_requests_give_identical_results(aclient):
    responses = await asyncio.gather(*(_ask(aclient, "atlas") for _ in range(REQUESTS)))
    bodies = [r.json() for r in responses]
    assert all(b == bodies[0] for b in bodies)
    chunk_ids = [s["chunk_id"] for s in bodies[0]["sources"]]
    assert len(chunk_ids) == len(set(chunk_ids))


async def test_different_questions_do_not_mix_evidence(aclient):
    questions = {
        "When are settlement files published?": "at2",
        "Where is chargeback evidence uploaded?": "at3",
        "What is the refund policy?": "at1",
    }
    jobs = list(questions.items()) * 5
    responses = await asyncio.gather(*(_ask(aclient, "atlas", q) for q, _ in jobs))

    for (_, expected_id), r in zip(jobs, responses, strict=True):
        assert r.json()["sources"][0]["chunk_id"] == expected_id


async def test_one_tenants_failure_does_not_affect_another(aclient, monkeypatch):
    real_search = vector_client.search

    async def atlas_is_down(query, tenant_id, top_k):
        if tenant_id == "atlas":
            raise ConnectionError("atlas shard down")
        return await real_search(query, tenant_id, top_k)

    monkeypatch.setattr(vector_client, "search", atlas_is_down)

    tenants = ["atlas", "cedar"] * 5
    responses = await asyncio.gather(*(_ask(aclient, t) for t in tenants))

    for tenant, r in zip(tenants, responses, strict=True):
        body = r.json()
        assert r.status_code == 200
        assert body["account_id"] == tenant
        assert body["degraded"] is (tenant == "atlas")
        assert bool(body["sources"]) is (tenant == "cedar")


async def test_concurrent_failures_are_bounded_in_time(aclient, monkeypatch):
    async def always_down(*args, **kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(vector_client, "search", always_down)

    started = time.perf_counter()
    responses = await asyncio.gather(*(_ask(aclient, "atlas") for _ in range(REQUESTS)))
    elapsed = time.perf_counter() - started

    assert all(r.json()["degraded"] is True for r in responses)
    assert elapsed < 1.0
