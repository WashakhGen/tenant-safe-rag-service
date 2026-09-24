import asyncio
import logging
import time

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.llm import model_client
from app.main import app
from app.vector import vector_client

client = TestClient(app)

TOTAL_ATTEMPTS = settings.max_retries + 1


def _ask(tenant="atlas", question="What is the refund policy?"):
    return client.post(
        "/v1/answer",
        headers={"X-Account-ID": tenant},
        json={"question": question},
    )


@pytest.fixture
def search_calls(monkeypatch):
    """Count every call made to the vector store, keeping its real behaviour."""
    calls = []
    real_search = vector_client.search

    async def counting_search(*args, **kwargs):
        calls.append(args)
        return await real_search(*args, **kwargs)

    monkeypatch.setattr(vector_client, "search", counting_search)
    return calls


@pytest.fixture
def broken_search(monkeypatch):
    """Make the vector store always raise, and count the calls."""
    calls = []

    def _install(error):
        async def failing_search(*args, **kwargs):
            calls.append(args)
            raise error

        monkeypatch.setattr(vector_client, "search", failing_search)
        return calls

    return _install


# retry recovers


def test_single_failure_is_retried_and_hidden_from_caller(monkeypatch, search_calls):
    monkeypatch.setattr(vector_client, "fail_next", True)
    r = _ask()
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is False
    assert body["sources"]
    assert "5 calendar days" in body["answer"]
    assert len(search_calls) == 2  # one failure, one successful retry


def test_healthy_dependency_is_called_once(search_calls):
    r = _ask()
    assert r.json()["degraded"] is False
    assert len(search_calls) == 1


#  retries are exhausted
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timeout"),
        ConnectionError("down"),
        RuntimeError("boom"),
        ValueError("bad response"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_persistent_failure_returns_degraded_response(broken_search, error):
    calls = broken_search(error)
    r = _ask()
    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["sources"] == []
    assert body["account_id"] == "atlas"
    assert "temporarily unavailable" in body["answer"]
    assert len(calls) == TOTAL_ATTEMPTS


def test_retries_are_bounded_by_configuration(monkeypatch, broken_search):
    monkeypatch.setattr(settings, "max_retries", 0)
    calls = broken_search(ConnectionError("down"))
    r = _ask()
    assert r.json()["degraded"] is True
    assert len(calls) == 1

    monkeypatch.setattr(settings, "max_retries", 4)
    calls.clear()
    _ask()
    assert len(calls) == 5


def test_degraded_response_does_not_call_the_model(monkeypatch, broken_search):
    broken_search(ConnectionError("down"))

    async def model_must_not_run(*args, **kwargs):
        raise AssertionError("model called without evidence")

    monkeypatch.setattr(model_client, "answer", model_must_not_run)
    assert _ask().json()["degraded"] is True


def test_degraded_response_is_logged(caplog, broken_search):
    broken_search(ConnectionError("down"))
    with caplog.at_level(logging.WARNING):
        _ask()
    assert any("vector store unavailable" in m for m in caplog.messages)
    assert sum("vector search failed" in m for m in caplog.messages) == TOTAL_ATTEMPTS


# slow dependency


def test_slow_dependency_is_cut_off_and_bounded(monkeypatch):
    monkeypatch.setattr(settings, "vector_timeout_seconds", 0.05)
    calls = []

    async def hanging_search(*args, **kwargs):
        calls.append(args)
        await asyncio.sleep(30)

    monkeypatch.setattr(vector_client, "search", hanging_search)

    started = time.perf_counter()
    r = _ask()
    elapsed = time.perf_counter() - started

    assert r.status_code == 200
    assert r.json()["degraded"] is True
    assert len(calls) == TOTAL_ATTEMPTS
    assert elapsed < TOTAL_ATTEMPTS * 0.05 + 1.0  # bounded, nowhere near 30s


def test_dependency_just_within_timeout_still_succeeds(monkeypatch):
    monkeypatch.setattr(settings, "vector_timeout_seconds", 1.0)
    monkeypatch.setattr(vector_client, "delay_seconds", 0.05)
    assert _ask().json()["degraded"] is False


# failure does not leak or stick


def test_failure_does_not_leak_into_the_next_request(monkeypatch, broken_search):
    broken_search(ConnectionError("down"))
    assert _ask("atlas").json()["degraded"] is True
    monkeypatch.undo()  # dependency recovers

    r = _ask("cedar")
    body = r.json()
    assert body["degraded"] is False
    assert body["account_id"] == "cedar"
    assert body["sources"]
    assert all(s["account_id"] == "cedar" for s in body["sources"])
    assert "10 calendar days" in body["answer"]


def test_failure_flag_is_consumed_after_one_failure(monkeypatch):
    monkeypatch.setattr(vector_client, "fail_next", True)
    _ask()
    assert vector_client.fail_next is False


def test_degraded_response_for_unknown_tenant_keeps_trusted_identity(broken_search):
    broken_search(ConnectionError("down"))
    r = _ask("nobody")
    assert r.status_code == 200
    assert r.json()["account_id"] == "nobody"
    assert r.json()["sources"] == []
