import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel, HttpUrl
from src.db.pinecone_client import PineconeVectorDB, get_vector_db
from src.rag.ingestion import DocumentIngestor
from src.rag.generator import RAGGenerator
from src.eval.ragas_evaluator import RagasEvaluator, get_evaluator

logger = logging.getLogger(__name__)

router = APIRouter()

class IngestRequest(BaseModel):
    text: str
    doc_id: str
    source_url: HttpUrl | None = None

class QueryRequest(BaseModel):
    question: str

@router.post("/ingest", summary="Ingest raw text into Pinecone")
def ingest_document(
    request: IngestRequest, 
    db: PineconeVectorDB = Depends(get_vector_db)
):
    """
    Takes a raw text payload, semantic chunks it via LangChain,
    vectorizes it using HuggingFace sentence-transformers,
    and upserts it into Pinecone.
    """
    logger.info(f"Received ingestion request for doc: {request.doc_id}")
    
    ingestor = DocumentIngestor(db_client=db)
    
    metadata = {
        "doc_id": request.doc_id,
        "source_url": str(request.source_url) if request.source_url else "unknown"
    }
    
    result = ingestor.ingest_text(request.text, metadata)
    return result

@router.post("/ask", summary="Perform a RAG query and evaluate it")
def query_rag(
    request: QueryRequest,
    db: PineconeVectorDB = Depends(get_vector_db),
    evaluator: RagasEvaluator = Depends(get_evaluator)
):
    """
    Takes a user question, retrieves relevant context from Pinecone,
    generates an answer via LLaMA-3, and runs Ragas assessment metrics.
    """
    logger.info(f"Received RAG query: {request.question}")
    
    # 1. Retrieval & Generation
    generator = RAGGenerator(db_client=db)
    rag_result = generator.generate_answer(request.question)
    
    answer = rag_result.get("answer", "")
    contexts = rag_result.get("contexts", [])
    
    # 2. Evaluation
    logger.info("Proceeding to Ragas evaluation...")
    metrics = evaluator.evaluate_response(request.question, answer, contexts)
    
    return {
        "question": request.question,
        "answer": answer,
        "contexts_retrieved": len(contexts),
        "evaluation_metrics": metrics
    }
