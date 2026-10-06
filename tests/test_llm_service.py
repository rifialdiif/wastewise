"""Phase 6 checks: Gemini recommender behaviour, with the Gemini client faked out."""

import json
import os
from types import SimpleNamespace

import pytest
from google.genai import errors

from app.config import settings
from app.services.llm_service import GeminiRecommender, LLMServiceError
from app.services.recommendation import KnowledgeBase

KB = KnowledgeBase(settings.resolve(settings.kb_path), ["plastic", "trash"])
VALID_JSON = json.dumps({"summary": "Recycle it.", "steps": ["Empty it."], "warnings": ["Check locally."]})


class FakeModels:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return SimpleNamespace(text=self.result)


def make_recommender(result) -> tuple[GeminiRecommender, FakeModels]:
    recommender = GeminiRecommender(api_key="test-key", model="test-model")
    fake = FakeModels(result)
    recommender.client = SimpleNamespace(models=fake)
    return recommender, fake


def test_missing_api_key_raises():
    with pytest.raises(LLMServiceError, match="GEMINI_API_KEY"):
        GeminiRecommender(api_key="", model="test-model")


def test_prompt_contains_only_class_confidence_status_and_kb_entry():
    prompt = json.loads(GeminiRecommender.build_prompt("plastic", 0.9612, KB.get("plastic")))
    assert prompt == {
        "predicted_class": "plastic",
        "confidence": 0.9612,
        "prediction_status": "accepted",
        "knowledge_base_entry": KB.get("plastic"),
    }


def test_uncertain_prompt_is_flagged_and_instructed():
    prompt = json.loads(GeminiRecommender.build_prompt("glass", 0.66, KB.get("plastic"), uncertain=True))
    assert prompt["prediction_status"] == "uncertain"

    recommender, fake = make_recommender(VALID_JSON)
    recommender.recommend("plastic", 0.66, KB.get("plastic"), uncertain=True)
    config = fake.calls[0]["config"]
    assert "Never present an uncertain class as confirmed" in config.system_instruction
    assert '"prediction_status": "uncertain"' in fake.calls[0]["contents"]


def test_valid_response_is_parsed_with_structured_output():
    recommender, fake = make_recommender(VALID_JSON)
    recommendation = recommender.recommend("plastic", 0.9612, KB.get("plastic"))

    assert recommendation.steps == ["Empty it."]
    config = fake.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema.__name__ == "Recommendation"
    assert "Use ONLY the facts" in config.system_instruction


@pytest.mark.parametrize(
    "failure",
    [
        errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
        errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}),
        TimeoutError("timed out"),
    ],
    ids=["quota", "server", "timeout"],
)
def test_api_failures_raise_llm_service_error(failure):
    recommender, _ = make_recommender(failure)
    with pytest.raises(LLMServiceError):
        recommender.recommend("plastic", 0.95, KB.get("plastic"))


@pytest.mark.parametrize(
    "text",
    [None, "not json", json.dumps({"summary": "x"}), json.dumps({"summary": " ", "steps": [], "warnings": []})],
    ids=["empty", "prose", "missing-fields", "blank"],
)
def test_invalid_output_raises_llm_service_error(text):
    recommender, _ = make_recommender(text)
    with pytest.raises(LLMServiceError):
        recommender.recommend("plastic", 0.95, KB.get("plastic"))


@pytest.mark.skipif(os.getenv("RUN_LIVE_GEMINI") != "1", reason="set RUN_LIVE_GEMINI=1 to call the real API")
def test_live_gemini_recommendation():
    recommender = GeminiRecommender(settings.gemini_api_key, settings.gemini_model)
    recommendation = recommender.recommend("trash", 0.93, KB.get("trash"))
    assert recommendation.summary and recommendation.steps
