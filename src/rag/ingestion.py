import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from src.db.pinecone_client import PineconeVectorDB

logger = logging.getLogger(__name__)

class DocumentIngestor:
    def __init__(self, db_client: PineconeVectorDB):
        self.db = db_client
        # You could also move the model name to settings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def ingest_text(self, document_text: str, source_metadata: dict = None) -> dict:
        """
        Chunks the input text, generates embeddings, and upserts to Pinecone.
        """
        if not source_metadata:
            source_metadata = {}

        logger.info(f"Starting ingestion for text of length: {len(document_text)}")
        
        # 1. Chunking
        chunks = self.text_splitter.split_text(document_text)
        logger.info(f"Generated {len(chunks)} chunks.")

        # 2. Embedding
        # If running in dummy mode without real settings, we might want to bypass real embedding to save time,
        # but for a production codebase we assume the model loads efficiently.
        vectors_to_upsert = []
        
        for i, chunk in enumerate(chunks):
            # Generate embedding vector
            vector = self.embeddings.embed_query(chunk)
            
            # Prepare metadata for Pinecone
            metadata = source_metadata.copy()
            metadata["text"] = chunk
            metadata["chunk_index"] = i
            
            # Unique ID for the vector
            vector_id = f"{source_metadata.get('doc_id', 'doc')}_chunk_{i}"
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": vector,
                "metadata": metadata
            })

        # 3. Upserting
        result = self.db.upsert(vectors_to_upsert)
        logger.info(f"Ingestion complete. Upserted details: {result}")
        
        return {
            "status": "success",
            "chunks_processed": len(chunks),
            "db_response": result
        }
