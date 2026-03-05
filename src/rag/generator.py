import logging
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from src.core.config import get_settings
from src.db.pinecone_client import PineconeVectorDB

logger = logging.getLogger(__name__)

class RAGGenerator:
    def __init__(self, db_client: PineconeVectorDB):
        self.db = db_client
        self.settings = get_settings()
        
        # We simulate a production endpoint using HuggingFace's Inference API
        # In a real environment, this could point to a local vLLM, TGI server,
        # or a managed cloud endpoint.
        try:
            self.llm = HuggingFaceEndpoint(
                repo_id=self.settings.hf_repo_id,
                task="text-generation",
                max_new_tokens=512,
                top_k=50,
                temperature=0.1,
                repetition_penalty=1.03,
                huggingfacehub_api_token=self.settings.huggingface_api_key
            )
        except Exception as e:
            logger.warning(f"Could not load HuggingFaceEndpoint. Missing token? Error: {e}")
            self.llm = None

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "You are an expert AI assistant. Answer the question relying ONLY on the provided Context. "
                "If you don't know the answer based on the Context, say 'I don't have enough information from the context.'\n"
                "Context:\n{context}<|eot_id|>\n"
                "<|start_header_id|>user<|end_header_id|>\n"
                "{question}<|eot_id|>\n"
                "<|start_header_id|>assistant<|end_header_id|>"
            )
        )
        
    def _create_fallback_response(self, question: str, contexts: list[str]) -> str:
        return (
            "⚠️ LLaMA-3 Endpoint is not configured (missing HF Token).\n\n"
            f"**Your Question:** {question}\n\n"
            "**Retrieved Contexts:**\n" + "\n---\n".join(contexts)
        )

    def generate_answer(self, question: str, top_k: int = 3) -> dict:
        """
        Runs the RAG logic: Retrieves context from Pinecone and generates an answer.
        """
        # 1. We must embed the user's question first to search Pinecone
        # For simplicity in this demo, we'll initialize a local embedder model
        # just for queries (in production, use a shared service or same pipeline)
        from langchain_huggingface import HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        query_vector = embedder.embed_query(question)
        
        # 2. Retrieve Document Context
        search_results = self.db.query(vector=query_vector, top_k=top_k)
        contexts = []
        
        if search_results.get("matches"):
            for match in search_results["matches"]:
                metadata = match.get("metadata", {})
                if "text" in metadata:
                    contexts.append(metadata["text"])

        context_string = "\n\n".join(contexts) if contexts else "No relevant context found."
        
        # 3. Generate Answer
        if self.llm is None:
            # Fallback for when API keys are not provided
            answer = self._create_fallback_response(question, contexts)
        else:
            prompt = self.prompt_template.format(context=context_string, question=question)
            try:
                answer = self.llm.invoke(prompt)
            except Exception as e:
                logger.error(f"Generation error: {e}")
                answer = f"Error generating answer: {e}"

        return {
            "answer": answer,
            "contexts": contexts
        }
