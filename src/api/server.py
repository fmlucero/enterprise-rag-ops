from fastapi import FastAPI
from src.core.config import get_settings
from src.api.routers.rag_router import router as rag_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Production-ready Retrieval-Augmented Generation API with Evaluation metrics",
    version="1.0.0"
)

# Incorporate the RAG endpoints
app.include_router(rag_router, prefix="/api/v1")

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "environment": settings.environment,
        "service": settings.app_name
    }
