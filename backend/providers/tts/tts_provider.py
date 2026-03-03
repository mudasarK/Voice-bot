from backend.core.config import settings
from backend.core.interfaces import TTSProvider

def get_tts_provider() -> TTSProvider:
    provider = settings.TTS_PROVIDER.lower()
    if provider == "edge_tts":
        from backend.providers.tts.local_tts import EdgeTTSProvider
        return EdgeTTSProvider()
    else:
        raise ValueError(f"Unknown TTS Provider: {provider}")
