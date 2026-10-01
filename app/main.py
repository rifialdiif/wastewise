from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="WasteWise API",
    description="AI-based waste classification and treatment recommendation API.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok"}
