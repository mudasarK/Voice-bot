from backend.core.config import settings
from backend.core.interfaces import VectorStoreProvider, EmbeddingsProvider

def get_vector_store_provider(embeddings: EmbeddingsProvider) -> VectorStoreProvider:
    provider = settings.VECTOR_STORE_PROVIDER.lower()
    if provider == "pinecone":
        from backend.providers.vectorstore.pinecone_store import PineconeStoreProvider
        return PineconeStoreProvider(embeddings_provider=embeddings, index_name=settings.PINECONE_INDEX_NAME)
    else:
        raise ValueError(f"Unknown Vector Store Provider: {provider}")
