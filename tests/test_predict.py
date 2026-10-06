"""Phase 8 checks: POST /predict end to end, with Gemini faked out."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes import (
    RECOMMENDATION_UNAVAILABLE_MESSAGE,
    UNCERTAIN_MESSAGE,
    UNCERTAIN_UNAVAILABLE_MESSAGE,
)
from app.config import settings
from app.main import app
from app.schemas.prediction import Recommendation
from app.services.llm_service import LLMServiceError

FIXTURES = Path(__file__).parent / "fixtures"
RECOMMENDATION = Recommendation(summary="Recycle it.", steps=["Empty it."], warnings=["Check locally."])


class FakeRecommender:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []

    def recommend(self, class_name, confidence, kb_entry, uncertain=False):
        self.calls.append((class_name, confidence, kb_entry, uncertain))
        if self.fail:
            raise LLMServiceError("simulated Gemini outage")
        return RECOMMENDATION.model_copy(deep=True)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def recommender(monkeypatch):
    fake = FakeRecommender()
    monkeypatch.setattr(app.state, "recommender", fake)
    return fake


def upload(client, name="plastic.jpg", content=None, content_type="image/jpeg"):
    data = content if content is not None else (FIXTURES / name).read_bytes()
    return client.post("/predict", files={"file": (name, data, content_type)})


def test_accepted_prediction_includes_recommendation(client, recommender):
    response = upload(client, "plastic.jpg")

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["class"] == "plastic"
    assert body["prediction"]["status"] == "accepted"
    assert body["prediction"]["confidence"] >= settings.confidence_threshold
    assert body["recommendation"] == RECOMMENDATION.model_dump()
    assert "message" not in body
    assert "candidates" not in body

    class_name, confidence, kb_entry, uncertain = recommender.calls[0]
    assert class_name == "plastic"
    assert confidence == body["prediction"]["confidence"]
    assert kb_entry == app.state.knowledge_base.get("plastic")
    assert uncertain is False


def test_uncertain_prediction_gets_flagged_top1_recommendation(client, recommender, monkeypatch):
    # glass.jpg scores ~0.98, so a 0.999 threshold makes it uncertain.
    monkeypatch.setattr(settings, "confidence_threshold", 0.999)
    response = upload(client, "glass.jpg")

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["class"] == "glass"
    assert body["prediction"]["status"] == "uncertain"
    assert body["message"] == UNCERTAIN_MESSAGE.format(class_name="glass")

    # Gemini is asked for the top-1 class, told the prediction is uncertain.
    class_name, _, kb_entry, uncertain = recommender.calls[0]
    assert (class_name, uncertain) == ("glass", True)
    assert kb_entry == app.state.knowledge_base.get("glass")

    # The code (not the LLM) puts the verification warning first.
    warnings = body["recommendation"]["warnings"]
    percent = int(body["prediction"]["confidence"] * 100)
    assert warnings[0].startswith(f"Classification is uncertain ({percent}% confidence)")
    assert "glass" in warnings[0]
    assert warnings[1:] == RECOMMENDATION.warnings
    assert RECOMMENDATION.warnings == ["Check locally."]  # shared object not mutated

    candidates = body["candidates"]
    assert len(candidates) == 3
    assert candidates[0] == {"class": "glass", "confidence": body["prediction"]["confidence"]}
    confidences = [c["confidence"] for c in candidates]
    assert confidences == sorted(confidences, reverse=True)
    assert len({c["class"] for c in candidates}) == 3


def test_gemini_failure_still_returns_classification(client, monkeypatch):
    monkeypatch.setattr(app.state, "recommender", FakeRecommender(fail=True))
    response = upload(client, "metal.jpg")

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["class"] == "metal"
    assert body["prediction"]["status"] == "accepted"
    assert body["recommendation"] is None
    assert body["message"] == RECOMMENDATION_UNAVAILABLE_MESSAGE


def test_uncertain_with_gemini_failure_keeps_candidates(client, monkeypatch):
    monkeypatch.setattr(settings, "confidence_threshold", 0.999)
    monkeypatch.setattr(app.state, "recommender", FakeRecommender(fail=True))
    body = upload(client, "glass.jpg").json()

    assert body["prediction"]["status"] == "uncertain"
    assert body["recommendation"] is None
    assert body["message"] == UNCERTAIN_UNAVAILABLE_MESSAGE
    assert len(body["candidates"]) == 3


def test_missing_api_key_still_returns_classification(client, monkeypatch):
    monkeypatch.setattr(app.state, "recommender", None)
    body = upload(client, "paper.jpg").json()

    assert body["prediction"]["class"] == "paper"
    assert body["recommendation"] is None
    assert body["message"] == RECOMMENDATION_UNAVAILABLE_MESSAGE


def test_non_image_upload_returns_400(client, recommender):
    response = upload(client, "notes.txt", content=b"hello", content_type="text/plain")
    assert response.status_code == 400
    assert "not a supported image" in response.json()["detail"]


def test_missing_file_returns_422(client):
    assert client.post("/predict").status_code == 422


def test_model_not_loaded_returns_503(client, monkeypatch):
    monkeypatch.setattr(app.state, "classifier", None)
    response = upload(client, "plastic.jpg")
    assert response.status_code == 503
