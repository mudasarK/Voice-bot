"""FastAPI dependency injection for providers and services. Keeps routes modular and testable."""
from functools import lru_cache

from backend.core.config import settings
from backend.core.interfaces import (
    EmbeddingsProvider,
    LLMProvider,
    STTProvider,
    TTSProvider,
    VectorStoreProvider,
)
from backend.providers.embeddings.embeddings_provider import get_embeddings_provider
from backend.providers.llm.llm_provider import get_llm_provider
from backend.providers.stt.stt_provider import get_stt_provider
from backend.providers.tts.tts_provider import get_tts_provider
from backend.providers.vectorstore.vector_store import get_vector_store_provider
from backend.rag.chain import RAGChain
from backend.voice.processor import VoiceProcessor


@lru_cache(maxsize=1)
def _get_llm_provider() -> LLMProvider:
    return get_llm_provider()


@lru_cache(maxsize=1)
def _get_embeddings_provider() -> EmbeddingsProvider:
    return get_embeddings_provider()


@lru_cache(maxsize=1)
def _get_vector_store_provider(embeddings: EmbeddingsProvider) -> VectorStoreProvider:
    return get_vector_store_provider(embeddings)


@lru_cache(maxsize=1)
def _get_rag_chain(
    llm: LLMProvider,
    store: VectorStoreProvider,
) -> RAGChain:
    return RAGChain(llm, store)


@lru_cache(maxsize=1)
def _get_stt_provider() -> STTProvider:
    return get_stt_provider()


@lru_cache(maxsize=1)
def _get_tts_provider() -> TTSProvider:
    return get_tts_provider()


@lru_cache(maxsize=1)
def _get_voice_processor(
    stt: STTProvider,
    tts: TTSProvider,
    rag: RAGChain,
) -> VoiceProcessor:
    return VoiceProcessor(stt, tts, rag)


# FastAPI Depends() injectables
def get_llm() -> LLMProvider:
    return _get_llm_provider()


def get_embeddings() -> EmbeddingsProvider:
    return _get_embeddings_provider()


def get_vector_store() -> VectorStoreProvider:
    return _get_vector_store_provider(_get_embeddings_provider())


def get_rag() -> RAGChain:
    return _get_rag_chain(_get_llm_provider(), get_vector_store())


def get_stt() -> STTProvider:
    return _get_stt_provider()


def get_tts() -> TTSProvider:
    return _get_tts_provider()


def get_voice_processor() -> VoiceProcessor:
    return _get_voice_processor(
        _get_stt_provider(),
        _get_tts_provider(),
        _get_rag(),
    )
