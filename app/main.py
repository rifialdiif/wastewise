import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings
from app.services.classifier import ModelLoadError, WasteClassifier
from app.services.llm_service import GeminiRecommender, LLMServiceError
from app.services.recommendation import KnowledgeBase, KnowledgeBaseError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-request logs from the Gemini SDK
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


API_DESCRIPTION = """
Upload a photo of a single waste item to get its material class and, when the
classifier is confident, practical handling guidance.

* **Classifier:** MobileNetV2 (transfer learning), 6 classes.
* **Confidence gate:** predictions below the threshold (default 0.90) are marked
  `uncertain` and get no recommendation.
* **Knowledge base:** the only source of treatment facts.
* **Gemini:** turns the knowledge base entry into a concise recommendation.

Recommendations are general guidance. Accepted materials and requirements vary
by local facility, so always check local rules.
"""

app = FastAPI(
    title="WasteWise API",
    description=API_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.middleware("http")
async def reject_oversized_requests(request: Request, call_next):
    """Reject uploads by their declared size before the body is received."""
    content_length = request.headers.get("content-length")
    # Allow some headroom for multipart boundaries and form headers.
    if content_length and content_length.isdigit() and int(content_length) > settings.max_upload_bytes + 64 * 1024:
        return JSONResponse(
            status_code=413,
            content={"detail": f"File is too large. Maximum upload size is {settings.max_upload_mb:g} MB."},
        )
    return await call_next(request)


# Added last so it is the outermost middleware: error responses (e.g. 413) also get CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log unexpected errors and return a generic message without internal details."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again later."})


@app.get("/health", tags=["system"], summary="Service and component status")
def health(request: Request) -> dict:
    """Report whether the classifier loaded and whether Gemini recommendations are enabled."""
    return {
        "status": "ok",
        "classifier_loaded": request.app.state.classifier is not None,
        "recommendations_enabled": request.app.state.recommender is not None,
    }
