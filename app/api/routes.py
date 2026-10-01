import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.config import settings
from app.schemas.prediction import Prediction, PredictionStatus, PredictResponse
from app.services.classifier import InvalidImageError, apply_confidence_gate
from app.services.llm_service import LLMServiceError

logger = logging.getLogger("wastewise")
router = APIRouter()

UNCERTAIN_MESSAGE = "Material classification is uncertain. Please verify the waste type before treatment."
RECOMMENDATION_UNAVAILABLE_MESSAGE = (
    "Classification succeeded, but the recommendation service is currently unavailable. "
    "Please try again later or follow your local waste-handling guidance."
)


@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(request: Request, file: UploadFile = File(..., description="Photo of a single waste item.")):
    """Classify a waste image and, when the prediction is confident, recommend how to handle it."""
    state = request.app.state
    if state.classifier is None:
        raise HTTPException(status_code=503, detail="Classification model is not available.")

    try:
        result = state.classifier.predict(file.file.read())
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    status = apply_confidence_gate(result.confidence, settings.confidence_threshold)
    prediction = Prediction(predicted_class=result.class_name, confidence=result.confidence, status=status)

    # Gemini is never called for uncertain predictions.
    if status is PredictionStatus.UNCERTAIN:
        return PredictResponse(prediction=prediction, message=UNCERTAIN_MESSAGE)

    if state.recommender is None:
        return PredictResponse(prediction=prediction, message=RECOMMENDATION_UNAVAILABLE_MESSAGE)

    kb_entry = state.knowledge_base.get(result.class_name)
    try:
        recommendation = state.recommender.recommend(result.class_name, result.confidence, kb_entry)
    except LLMServiceError as exc:
        logger.warning("Recommendation failed for '%s': %s", result.class_name, exc)
        return PredictResponse(prediction=prediction, message=RECOMMENDATION_UNAVAILABLE_MESSAGE)

    return PredictResponse(prediction=prediction, recommendation=recommendation)
