import asyncio
import tempfile
import os
import edge_tts
from backend.core.interfaces import TTSProvider

class EdgeTTSProvider(TTSProvider):
    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice
        
    def synthesize(self, text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            # Run in isolated loop to avoid conflicting with FastAPI's event loop (safe when called from thread)
            asyncio.run(communicate.save(temp_audio_path))
            with open(temp_audio_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
