import tempfile
import os
import whisper
from backend.core.interfaces import STTProvider

class LocalWhisperSTTProvider(STTProvider):
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)
        
    def transcribe(self, audio_bytes: bytes, audio_format: str = "wav") -> str:
        # Write bytes to temp file because whisper needs a file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
            
        try:
            result = self.model.transcribe(temp_audio_path)
            return result["text"].strip()
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
