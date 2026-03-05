from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Enterprise RAG Ops API")

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/rag")
def run_rag(request: QueryRequest):
    # This will be replaced with our actual RAG logic
    return {
        "query": request.query,
        "response": "This is a placeholder response. VectorDB integration pending.",
        "metrics": {"faithfulness_score": 0.0, "relevance_score": 0.0}
    }
