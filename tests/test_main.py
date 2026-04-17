# tests/test_main.py
# Smoke tests for Phase 3 — POST /analyze endpoint

from starlette.testclient import TestClient

# Import deferred to allow patching if model artifacts are missing during CI
from main import app

client = TestClient(app)


def test_analyze_returns_envelope():
    """API-01 + API-02: POST /analyze returns 200 with {type, data} envelope; type is prediction or insight."""
    resp = client.post("/analyze", json={"message": "3 bed 2 bath house in NridgHt"})
    assert resp.status_code == 200
    body = resp.json()
    assert "type" in body, f"Missing 'type' key in response: {body}"
    assert "data" in body, f"Missing 'data' key in response: {body}"
    assert body["type"] in ("prediction", "insight"), f"Unexpected type value: {body['type']}"


def test_analyze_no_overrides():
    """API-03: assumed_overrides omitted (null) does not cause a 422 or 500."""
    resp = client.post("/analyze", json={"message": "small condo downtown"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] in ("prediction", "insight")


def test_cors_header():
    """API-04: CORS header Access-Control-Allow-Origin is present on responses."""
    resp = client.post(
        "/analyze",
        json={"message": "2 bed ranch style"},
        headers={"Origin": "http://localhost:8501"},
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers, (
        f"CORS header missing. Headers received: {dict(resp.headers)}"
    )
