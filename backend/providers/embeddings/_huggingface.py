from backend.core.interfaces import EmbeddingsProvider
from backend.core.config import settings
from langchain_huggingface import HuggingFaceEmbeddings

class HuggingFaceEmbeddingsProvider(EmbeddingsProvider):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDINGS_MODEL
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        return self.embeddings
