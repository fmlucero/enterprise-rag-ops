from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Settings
    app_name: str = "Enterprise RAG Ops API"
    environment: str = "development"
    
    # Pinecone
    pinecone_api_key: str = ""
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "enterprise-rag-ops"
    
    # LLM Settings
    huggingface_api_key: str = ""
    hf_repo_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def get_settings() -> Settings:
    return Settings()
