from typing import List, Any
from backend.core.interfaces import VectorStoreProvider, EmbeddingsProvider
from backend.core.config import settings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

class PineconeStoreProvider(VectorStoreProvider):
    def __init__(self, embeddings_provider: EmbeddingsProvider, index_name: str = None):
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.embeddings = embeddings_provider.get_embeddings()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # We assume the index exists. Otherwise, create it here or via another script.
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=settings.PINECONE_API_KEY
        )
        
    def get_store(self) -> PineconeVectorStore:
        return self.vector_store
        
    def add_documents(self, documents: List[Any]):
        self.vector_store.add_documents(documents)
