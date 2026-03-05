# Enterprise RAG Ops (Vector DB & Evaluator) 🚀

This repository contains a production-ready **Retrieval-Augmented Generation (RAG)** API built with FastAPI, LangChain, Pinecone, and Ragas. 

It demonstrates how to transition from local proof-of-concept experimentation into a scalable, observable LLMOps deployment standard by integrating dependency injection, remote LLM endpoints, and quantitative hallucination metrics.

## 🌟 Key Features

### 1. Robust API Architecture
- **FastAPI with Dependency Injection:** Database connections (Pinecone) and large evaluation models (Ragas) are structured as injected dependencies (`Depends()`). This prevents memory leaks and ensures high concurrency.
- **Environment Management:** Utilizes `pydantic-settings` to robustly validate `.env` keys at startup.

### 2. Multi-stage Ingestion Pipeline
- Processes raw text using LangChain's `RecursiveCharacterTextSplitter`.
- Generates precise dense embeddings via `Sentence-Transformers`.
- Upserts vectorized chunks alongside dynamic source metadata into a **Pinecone Serverless Vector Database**.

### 3. Remote Inference Simulating vLLM
- Connects directly to a `HuggingFaceEndpoint` (LLaMA-3 8B Instruct).
- Simulates how enterprise environments keep inference endpoints decoupled from the RAG orchestration server.

### 4. Ragas Hallucination Metrics
Integrates the `ragas` library into the API response to quantitatively score LLM outputs based on:
- **Faithfulness:** Discards answers that hallucinate information not present in the retrieved Pinecone context.
- **Answer Relevance:** Evaluates if the generated answer directly addresses the user's initial prompt.

---

## 🏗️ Repository Structure

```text
├── Dockerfile                # Production container specification
├── requirements.txt          # Explicit pinning of LLMOps libraries
├── src/
│   ├── api/
│   │   ├── routers/
│   │   │   └── rag_router.py # POST /ingest and /ask logic
│   │   ├── server.py         # FastAPI App Entrypoint
│   │   └── test_server.py    # Pytest schema and endpoint tests
│   ├── core/
│   │   └── config.py         # Pydantic Settings
│   ├── db/
│   │   └── pinecone_client.py# VectorDB Wrapper
│   ├── eval/
│   │   └── ragas_evaluator.py# LLM-as-a-judge scoring
│   └── rag/
│       ├── generator.py      # LangChain HuggingFace generation
│       └── ingestion.py      # Text processing and embedding
```

## 🚀 How to Run Locally

1. **Clone & Install:**
   ```bash
   git clone https://github.com/fmlucero/enterprise-rag-ops.git
   cd enterprise-rag-ops
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Create a `.env` file at the root:
   ```env
   PINECONE_API_KEY="your-pinecone-key"
   PINECONE_INDEX_NAME="enterprise-rag-ops"
   HUGGINGFACE_API_KEY="your-hf-token"
   ```

3. **Start the API Server:**
   ```bash
   uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
   ```
   Navigate to `http://localhost:8000/docs` to test endpoints interactively via Swagger UI.

## 🐳 Running via Docker

To eliminate environment variances:
```bash
docker build -t enterprise-rag-ops .
docker run -p 8000:8000 --env-file .env enterprise-rag-ops
```
