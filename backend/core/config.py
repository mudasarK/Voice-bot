from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # API Keys
    OPENROUTER_API_KEY: str = ""
    PINECONE_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""

    # Provider Selection
    LLM_PROVIDER: str = "openrouter"
    VECTOR_STORE_PROVIDER: str = "pinecone"
    EMBEDDINGS_PROVIDER: str = "huggingface"
    STT_PROVIDER: str = "openrouter"  # openrouter or local_whisper
    TTS_PROVIDER: str = "edge_tts"

    # Models & configurations
    LLM_MODEL: str = "google/gemma-3n-e4b-it:free"
    STT_MODEL: str = "google/gemma-3n-e4b-it:free"
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    PINECONE_INDEX_NAME: str = "helpdesk-index"

settings = Settings()
