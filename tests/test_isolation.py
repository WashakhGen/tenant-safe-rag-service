import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.vector import vector_client

client = TestClient(app)


def _ask(tenant, question="What is the refund policy?", **extra):
    return client.post(
        "/v1/answer",
        headers={"X-Account-ID": tenant},
        json={"question": question, **extra},
    )


def _sources_belong_to(response, tenant):
    return all(s["account_id"] == tenant for s in response.json()["sources"])


# trusted tenant identity


def test_account_override_cannot_replace_trusted_tenant():
    r = _ask("atlas", account_override="cedar")
    assert r.status_code == 200
    body = r.json()
    assert body["account_id"] == "atlas"
    assert body["sources"]
    assert _sources_belong_to(r, "atlas")
    assert "5 calendar days" in body["answer"]
    assert "10 calendar days" not in body["answer"]


def test_account_override_matching_trusted_tenant_is_harmless():
    r = _ask("cedar", account_override="cedar")
    assert r.status_code == 200
    assert r.json()["account_id"] == "cedar"


def test_ignored_override_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="app.main"):
        _ask("atlas", account_override="cedar")
    assert any("ignoring account_override" in m for m in caplog.messages)


def test_matching_override_is_not_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="app.main"):
        _ask("atlas", account_override="atlas")
    assert not caplog.messages


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_trusted_tenant_header_is_rejected(value):
    r = client.post(
        "/v1/answer",
        headers={"X-Account-ID": value},
        json={"question": "refund policy"},
    )
    assert r.status_code == 400


# retrieval isolation


@pytest.mark.parametrize("tenant", ["atlas", "cedar"])
async def test_search_returns_only_the_requested_tenant(tenant):
    results = await vector_client.search("refund settlement", tenant, 5)
    assert results
    assert {r.account_id for r in results} == {tenant}


async def test_search_top_k_is_applied_after_tenant_filter():
    results = await vector_client.search("refund", "cedar", 5)
    assert [r.chunk_id for r in results] == ["cd1", "cd2"]


async def test_search_unknown_tenant_returns_nothing():
    assert await vector_client.search("refund", "nobody", 5) == []


def test_each_tenant_gets_its_own_answer():
    atlas = _ask("atlas").json()
    cedar = _ask("cedar").json()
    assert "5 calendar days" in atlas["answer"]
    assert "10 calendar days" in cedar["answer"]


def test_unknown_tenant_gets_no_sources_and_safe_answer():
    r = _ask("nobody")
    assert r.status_code == 200
    body = r.json()
    assert body["account_id"] == "nobody"
    assert body["sources"] == []
    assert "not have enough evidence" in body["answer"]


# request isolation


def test_previous_request_does_not_leak_into_next_tenant():
    _ask("atlas")
    r = _ask("cedar")
    assert r.json()["sources"]
    assert _sources_belong_to(r, "cedar")


def test_repeated_request_gives_identical_result():
    first = _ask("atlas").json()
    second = _ask("atlas").json()
    assert first == second
    assert len(second["sources"]) == len(first["sources"])
