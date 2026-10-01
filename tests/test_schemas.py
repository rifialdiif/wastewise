"""Phase 7 checks: response schemas serialize to the documented API contract."""

import pytest
from pydantic import ValidationError

from app.schemas.prediction import (
    Candidate,
    Prediction,
    PredictionStatus,
    PredictResponse,
    Recommendation,
)


def test_accepted_response_matches_contract():
    response = PredictResponse(
        prediction=Prediction(predicted_class="plastic", confidence=0.9612, status=PredictionStatus.ACCEPTED),
        recommendation=Recommendation(summary="s", steps=["a", "b"], warnings=["w"]),
    )
    assert response.model_dump(by_alias=True, mode="json") == {
        "prediction": {"class": "plastic", "confidence": 0.9612, "status": "accepted"},
        "recommendation": {"summary": "s", "steps": ["a", "b"], "warnings": ["w"]},
    }


def test_uncertain_response_matches_contract():
    message = "Material classification is uncertain. Please verify the waste type before treatment."
    response = PredictResponse(
        prediction=Prediction(predicted_class="glass", confidence=0.61, status=PredictionStatus.UNCERTAIN),
        recommendation=None,
        message=message,
        candidates=[
            Candidate(candidate_class="glass", confidence=0.61),
            Candidate(candidate_class="plastic", confidence=0.3),
        ],
    )
    assert response.model_dump(by_alias=True, mode="json") == {
        "prediction": {"class": "glass", "confidence": 0.61, "status": "uncertain"},
        "recommendation": None,
        "message": message,
        "candidates": [
            {"class": "glass", "confidence": 0.61},
            {"class": "plastic", "confidence": 0.3},
        ],
    }


def test_prediction_accepts_class_alias():
    prediction = Prediction.model_validate({"class": "metal", "confidence": 0.95, "status": "accepted"})
    assert prediction.predicted_class == "metal"


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_must_be_a_probability(confidence):
    with pytest.raises(ValidationError):
        Prediction(predicted_class="paper", confidence=confidence, status=PredictionStatus.ACCEPTED)


def test_unknown_status_is_rejected():
    with pytest.raises(ValidationError):
        Prediction(predicted_class="paper", confidence=0.5, status="maybe")


def test_recommendation_requires_all_fields():
    with pytest.raises(ValidationError):
        Recommendation.model_validate({"summary": "only a summary"})
