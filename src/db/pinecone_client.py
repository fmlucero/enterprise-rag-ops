import logging
from pinecone import Pinecone
from src.core.config import get_settings

logger = logging.getLogger(__name__)

class PineconeVectorDB:
    """
    A Pinecone wrapper structured to be used as a FastAPI dependency.
    """
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.pinecone_api_key
        self.index_name = self.settings.pinecone_index_name
        self.pc = None
        self.index = None
        
        self._initialize_connection()

    def _initialize_connection(self):
        if not self.api_key:
            logger.warning("PINECONE_API_KEY is missing. Operating in dummy/mock mode.")
            return

        try:
            self.pc = Pinecone(api_key=self.api_key)
            # You might want to check if the index exists first in a real setup
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Successfully connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")

    def query(self, vector: list[float], top_k: int = 5, include_metadata: bool = True):
        if self.index:
            return self.index.query(
                vector=vector, 
                top_k=top_k, 
                include_metadata=include_metadata
            )
        else:
            logger.warning("Mock query executed due to missing Pinecone configuration.")
            metadata = {
                "text": "This is dummy text from the fallback vector DB.", 
                "source": "mock"
            }
            return {"matches": [{"id": "mock-0", "score": 0.99, "metadata": metadata}]}

    def upsert(self, vectors: list[dict]):
        if self.index:
            return self.index.upsert(vectors=vectors)
        else:
            logger.warning(f"Mock upsert of {len(vectors)} vectors executed due to missing Pinecone configuration.")
            return {"upserted_count": len(vectors)}

# Dependency injector instance
db_client = PineconeVectorDB()

def get_vector_db() -> PineconeVectorDB:
    """
    FastAPI dependency injection provider.
    """
    return db_client
