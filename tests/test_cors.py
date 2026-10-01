"""CORS checks: only configured website origins may call the API from a browser."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, settings
from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def preflight(client, origin: str):
    return client.options(
        "/predict",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )


def test_origins_are_parsed_from_comma_separated_list():
    assert settings.cors_origin_list == ["https://allowed.example", "http://localhost:5173"]
    assert Settings(cors_origins="").cors_origin_list == []


def test_allowed_origin_passes_preflight(client):
    response = preflight(client, "https://allowed.example")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://allowed.example"


def test_unknown_origin_is_not_allowed(client):
    response = preflight(client, "https://evil.example")
    assert "access-control-allow-origin" not in response.headers


def test_error_responses_include_cors_headers(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0.005)
    response = client.post(
        "/predict",
        headers={"Origin": "http://localhost:5173"},
        files={"file": ("big.jpg", b"\0" * 200_000, "image/jpeg")},
    )
    assert response.status_code == 413
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
