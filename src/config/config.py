import os
from dataclasses import dataclass


@dataclass
class Settings:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    uploads_dir: str = os.getenv("UPLOADS_DIR", "data/uploads")
    vector_store_dir: str = os.getenv("VECTOR_STORE_DIR", "data/vector_store")
    google_sheet_id: str = os.getenv("GOOGLE_SHEET_ID", "")
    google_service_account_json: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "credentials.json")

    chunk_size: int = 1000
    chunk_overlap: int = 150

    retrieval_k: int = 4

    memory_mode: str = os.getenv("MEMORY_MODE", "buffer")
    summary_trigger_turns: int = 8  # only used when memory_mode == "summary_buffer"


def get_settings() -> Settings:
    return Settings()
