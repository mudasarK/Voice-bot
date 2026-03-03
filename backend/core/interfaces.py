from abc import ABC, abstractmethod
from typing import Any, List, Dict

class LLMProvider(ABC):
    @abstractmethod
    def get_llm(self) -> Any:
        pass

class VectorStoreProvider(ABC):
    @abstractmethod
    def get_store(self) -> Any:
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Any]):
        pass

class EmbeddingsProvider(ABC):
    @abstractmethod
    def get_embeddings(self) -> Any:
        pass

class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, audio_format: str = "wav") -> str:
        """Transcribe audio bytes to text. audio_format is e.g. 'wav', 'mp3' (provider may ignore)."""
        pass

class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes."""
        pass
