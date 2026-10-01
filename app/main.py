import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.routes import router
from app.config import settings
from app.services.classifier import ModelLoadError, WasteClassifier
from app.services.llm_service import GeminiRecommender, LLMServiceError
from app.services.recommendation import KnowledgeBase, KnowledgeBaseError

logger = logging.getLogger("wastewise")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model, knowledge base and Gemini client once at startup.

    A failure is logged and leaves the component as None, so the API can still
    start and report the problem instead of crashing.
    """
    app.state.classifier = None
    app.state.knowledge_base = None
    app.state.recommender = None

    try:
        app.state.classifier = WasteClassifier(
            settings.resolve(settings.model_path), settings.resolve(settings.class_mapping_path)
        )
        app.state.knowledge_base = KnowledgeBase(
            settings.resolve(settings.kb_path), app.state.classifier.class_names
        )
    except (ModelLoadError, KnowledgeBaseError) as exc:
        logger.error("Classifier unavailable: %s", exc)

    try:
        app.state.recommender = GeminiRecommender(settings.gemini_api_key, settings.gemini_model)
    except LLMServiceError as exc:
        logger.warning("Recommendations disabled: %s", exc)

    yield


app = FastAPI(
    title="WasteWise API",
    description="AI-based waste classification and treatment recommendation API.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health", tags=["system"])
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "classifier_loaded": request.app.state.classifier is not None,
        "recommendations_enabled": request.app.state.recommender is not None,
    }
