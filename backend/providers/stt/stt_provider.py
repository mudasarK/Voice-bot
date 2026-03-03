from backend.core.config import settings
from backend.core.interfaces import STTProvider

def get_stt_provider() -> STTProvider:
    provider = settings.STT_PROVIDER.lower()
    if provider == "openrouter":
        from backend.providers.stt.openrouter_stt import OpenRouterSTTProvider
        return OpenRouterSTTProvider(model_name=settings.STT_MODEL)
    elif provider == "local_whisper":
        from backend.providers.stt.local_whisper import LocalWhisperSTTProvider
        return LocalWhisperSTTProvider(model_size="base")
    else:
        raise ValueError(f"Unknown STT Provider: {provider}")
