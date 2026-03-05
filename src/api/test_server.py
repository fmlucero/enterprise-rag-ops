from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ask_endpoint_schema():
    # We test the schema validation
    # Expect 422 Unprocessable Entity if question is missing
    response = client.post("/api/v1/ask", json={})
    assert response.status_code == 422
    
def test_ingest_endpoint_schema():
    # Expect 422 if text and doc_id are missing
    response = client.post("/api/v1/ingest", json={})
    assert response.status_code == 422
