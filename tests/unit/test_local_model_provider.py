from __future__ import annotations

from fastapi.testclient import TestClient

from harness.local_model_provider import build_app


def test_local_model_provider_builds_and_serves_health() -> None:
    app = build_app(model="local-foundation:v1", backend="fallback", max_tokens=32)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["service"] == "local_model_provider"


def test_local_model_provider_builds_models_payload() -> None:
    app = build_app(model="local-foundation:v1", backend="fallback")
    client = TestClient(app)

    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"] and isinstance(payload["data"], list)
    assert payload["data"][0]["id"] == "local-foundation:v1"
