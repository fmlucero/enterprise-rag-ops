import os
from pinecone import Pinecone

class VectorDB:
    def __init__(self):
        # We will initialize this strictly with env variables to follow 12-factor apps
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENV")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "enterprise-rag")
        
        if self.api_key:
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
        else:
            print("WARNING: PINECONE_API_KEY not set. Operating in dummy mode.")
            self.pc = None
            self.index = None

    def search(self, vector: list, top_k: int = 5):
        if self.index:
            return self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        return {"matches": [{"metadata": {"text": "dummy context due to no api key"}}]}

