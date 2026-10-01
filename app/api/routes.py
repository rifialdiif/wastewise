import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.config import settings
from app.schemas.prediction import (
    Candidate,
    ErrorResponse,
    Prediction,
    PredictionStatus,
    PredictResponse,
)
from app.services.classifier import InvalidImageError, apply_confidence_gate
from app.services.llm_service import LLMServiceError

logger = logging.getLogger("wastewise")
router = APIRouter()

UNCERTAIN_MESSAGE = "Material classification is uncertain. Please verify the waste type before treatment."
RECOMMENDATION_UNAVAILABLE_MESSAGE = (
    "Classification succeeded, but the recommendation service is currently unavailable. "
    "Please try again later or follow your local waste-handling guidance."
)


def _error(description: str, detail: str) -> dict:
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {"application/json": {"example": {"detail": detail}}},
    }


PREDICT_RESPONSES = {
    200: {
        "description": "Classification result, with a recommendation when the prediction is accepted.",
        "content": {
            "application/json": {
                "examples": {
                    "accepted": {
                        "summary": "Confident prediction with recommendation",
                        "value": {
                            "prediction": {"class": "plastic", "confidence": 0.9919, "status": "accepted"},
                            "recommendation": {
                                "summary": "Plastic items should be reused if clean and suitable, or separated "
                                "for recycling where appropriate collection facilities exist.",
                                "steps": [
                                    "Empty all remaining contents from the item.",
                                    "Remove non-plastic components when practical.",
                                    "Separate for recycling where a suitable collection facility exists.",
                                ],
                                "warnings": [
                                    "Do not burn plastic waste in open areas.",
                                    "Check with your local collection or recycling facility, as acceptance "
                                    "depends on the plastic type.",
                                ],
                            },
                        },
                    },
                    "uncertain": {
                        "summary": "Confidence below threshold: no recommendation, top-3 candidates instead",
                        "value": {
                            "prediction": {"class": "glass", "confidence": 0.6641, "status": "uncertain"},
                            "recommendation": None,
                            "message": UNCERTAIN_MESSAGE,
                            "candidates": [
                                {"class": "glass", "confidence": 0.6641},
                                {"class": "plastic", "confidence": 0.2769},
                                {"class": "paper", "confidence": 0.0429},
                            ],
                        },
                    },
                    "recommendation_unavailable": {
                        "summary": "Accepted, but Gemini failed or is not configured",
                        "value": {
                            "prediction": {"class": "metal", "confidence": 0.9933, "status": "accepted"},
                            "recommendation": None,
                            "message": RECOMMENDATION_UNAVAILABLE_MESSAGE,
                        },
                    },
                }
            }
        },
    },
    400: _error(
        "Empty, corrupted or unsupported file.",
        "File is not a supported image. Please upload a JPEG, PNG, WebP or BMP photo.",
    ),
    413: _error("Upload exceeds MAX_UPLOAD_MB.", "File is too large. Maximum upload size is 10 MB."),
    500: _error("Unexpected server error.", "Internal server error. Please try again later."),
    503: _error("The classification model failed to load.", "Classification model is not available."),
}


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses=PREDICT_RESPONSES,
    tags=["prediction"],
    summary="Classify a waste photo and recommend how to handle it",
)
def predict(
    request: Request,
    file: UploadFile = File(..., description="Photo of a single waste item (JPEG, PNG, WebP or BMP)."),
):
    """Classify the material in a photo and, if the prediction is confident, explain how to handle it.

    1. **Classify** with MobileNetV2 into cardboard, glass, metal, paper, plastic or trash.
    2. **Gate** on confidence (`CONFIDENCE_THRESHOLD`, default 0.90). Below it, the status is
       `uncertain`, no recommendation is generated and the top-3 `candidates` are returned as hints.
    3. **Recommend** (accepted only): Gemini rephrases the knowledge base entry for the class.
       It may not add treatment rules beyond that entry.

    If Gemini fails, the classification is still returned with an explanatory `message`.
    """
    state = request.app.state
    if state.classifier is None:
        raise HTTPException(status_code=503, detail="Classification model is not available.")

    # Read at most one byte past the limit, so oversized uploads are never fully loaded.
    image_bytes = file.file.read(settings.max_upload_bytes + 1)
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413, detail=f"File is too large. Maximum upload size is {settings.max_upload_mb:g} MB."
        )

    try:
        result = state.classifier.predict(image_bytes)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    status = apply_confidence_gate(result.confidence, settings.confidence_threshold)
    prediction = Prediction(predicted_class=result.class_name, confidence=result.confidence, status=status)

    # Gemini is never called for uncertain predictions; the top candidates help the user verify.
    if status is PredictionStatus.UNCERTAIN:
        candidates = [
            Candidate(candidate_class=name, confidence=confidence) for name, confidence in result.top_k
        ]
        return PredictResponse(prediction=prediction, message=UNCERTAIN_MESSAGE, candidates=candidates)

    if state.recommender is None:
        return PredictResponse(prediction=prediction, message=RECOMMENDATION_UNAVAILABLE_MESSAGE)

    kb_entry = state.knowledge_base.get(result.class_name)
    try:
        recommendation = state.recommender.recommend(result.class_name, result.confidence, kb_entry)
    except LLMServiceError as exc:
        logger.warning("Recommendation failed for '%s': %s", result.class_name, exc)
        return PredictResponse(prediction=prediction, message=RECOMMENDATION_UNAVAILABLE_MESSAGE)

    return PredictResponse(prediction=prediction, recommendation=recommendation)
