from backend.core.config import settings
from backend.core.interfaces import EmbeddingsProvider

def get_embeddings_provider() -> EmbeddingsProvider:
    provider = settings.EMBEDDINGS_PROVIDER.lower()
    if provider == "huggingface":
        from backend.providers.embeddings._huggingface import HuggingFaceEmbeddingsProvider
        return HuggingFaceEmbeddingsProvider(model_name=settings.EMBEDDINGS_MODEL)
    else:
        raise ValueError(f"Unknown Embeddings Provider: {provider}")
