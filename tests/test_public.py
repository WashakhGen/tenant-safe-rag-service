from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_missing_trusted_tenant_header_is_rejected():
    r = client.post("/v1/answer", json={"question": "refund policy"})
    assert r.status_code == 400


def test_answer_endpoint_is_reachable():
    r = client.post(
        "/v1/answer",
        headers={"X-Account-ID": "atlas"},
        json={"question": "refund policy"},
    )
    assert r.status_code == 200
    assert r.json()["account_id"] == "atlas"
