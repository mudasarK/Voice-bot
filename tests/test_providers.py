"""Unit tests: verify each provider can be instantiated (with mocked or minimal config)."""
import os
import pytest


@pytest.fixture(autouse=True)
def _backend_cwd():
    """Ensure we can import backend from Voice-bot root."""
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def test_llm_provider_openrouter_instantiation():
    """OpenRouter LLM provider can be instantiated when API key is set or empty."""
    from backend.providers.llm.openrouter import OpenRouterLLMProvider
    from backend.core.config import settings
    # May have empty key in CI; provider still constructs
    p = OpenRouterLLMProvider(model_name=settings.LLM_MODEL)
    assert p.get_llm() is not None


def test_embeddings_provider_huggingface_instantiation():
    """HuggingFace embeddings provider can be instantiated."""
    pytest.importorskip("langchain_huggingface")
    from backend.providers.embeddings._huggingface import HuggingFaceEmbeddingsProvider
    from backend.core.config import settings
    p = HuggingFaceEmbeddingsProvider(model_name=settings.EMBEDDINGS_MODEL)
    assert p.get_embeddings() is not None


def test_stt_provider_openrouter_instantiation():
    """OpenRouter STT provider can be instantiated; empty audio + no key returns ''."""
    from backend.core.config import settings
    from backend.providers.stt.openrouter_stt import OpenRouterSTTProvider
    p = OpenRouterSTTProvider(model_name=settings.STT_MODEL)
    # With empty bytes and/or missing API key, transcribe returns ""
    out = p.transcribe(b"", audio_format="wav")
    assert isinstance(out, str)


@pytest.mark.skip(reason="Requires whisper model download; run manually if needed")
def test_stt_provider_local_whisper_transcribe():
    """LocalWhisper transcribes audio (slow, requires model)."""
    pytest.importorskip("whisper")
    from backend.providers.stt.local_whisper import LocalWhisperSTTProvider
    p = LocalWhisperSTTProvider(model_size="tiny")
    out = p.transcribe(b"\x00" * 8000, audio_format="wav")
    assert isinstance(out, str)


def test_tts_provider_edge_instantiation():
    """Edge TTS provider can be instantiated."""
    pytest.importorskip("edge_tts")
    from backend.providers.tts.local_tts import EdgeTTSProvider
    p = EdgeTTSProvider(voice="en-US-AriaNeural")
    assert p.voice == "en-US-AriaNeural"


def test_tts_provider_synthesize_returns_bytes():
    """Edge TTS synthesize returns bytes."""
    pytest.importorskip("edge_tts")
    from backend.providers.tts.local_tts import EdgeTTSProvider
    p = EdgeTTSProvider(voice="en-US-AriaNeural")
    out = p.synthesize("Hello")
    assert isinstance(out, bytes)
    assert len(out) > 0


def test_document_ingestor_load_and_split_txt(tmp_path):
    """DocumentIngestor loads and splits a text file."""
    from backend.rag.document_loader import DocumentIngestor
    f = tmp_path / "doc.txt"
    f.write_text("First chunk here. " * 50 + "\n\nSecond chunk there. " * 50)
    ingestor = DocumentIngestor(chunk_size=100, chunk_overlap=20)
    docs = ingestor.load_and_split(str(f))
    assert len(docs) >= 1
    assert "First chunk" in docs[0].page_content or "Second chunk" in docs[0].page_content


def test_voice_processor_process_audio_empty_returns_fallback(monkeypatch):
    """VoiceProcessor returns fallback when STT returns empty."""
    from backend.voice.processor import VoiceProcessor
    from backend.core.interfaces import STTProvider, TTSProvider

    class EmptySTT(STTProvider):
        def transcribe(self, audio_bytes: bytes, audio_format: str = "wav") -> str:
            return ""

    class EchoTTS(TTSProvider):
        def synthesize(self, text: str) -> bytes:
            return text.encode("utf-8")

    class EchoRAG:
        def invoke(self, question: str) -> str:
            return "Echo: " + question

    stt, tts, rag = EmptySTT(), EchoTTS(), EchoRAG()
    processor = VoiceProcessor(stt, tts, rag)
    user_text, bot_text, audio_bytes = processor.process_audio(b"\x00\x00", "wav")
    assert user_text == ""
    assert "couldn't hear" in bot_text.lower()
    assert len(audio_bytes) > 0
