import base64
import requests
import json
from backend.core.interfaces import STTProvider
from backend.core.config import settings
import logging

class OpenRouterSTTProvider(STTProvider):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.STT_MODEL
        self.api_key = settings.OPENROUTER_API_KEY
        
    def transcribe(self, audio_bytes: bytes, audio_format: str = "wav") -> str:
        """Transcribe audio bytes to text using OpenRouter's input_audio format."""
        if not self.api_key:
            logging.error("OpenRouter API key is not set for STT!")
            return ""

        logging.info(f"Using OpenRouter model {self.model_name} for Speech-to-Text")
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Help Desk Voice Bot STT",
            "Content-Type": "application/json",
        }

        # OpenRouter expects type "input_audio" with data + format (see openrouter.ai/docs/guides/overview/multimodal/audio)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe the following audio accurately. Return ONLY the transcribed text, nothing else.",
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_base64,
                                "format": audio_format,
                            },
                        },
                    ],
                }
            ],
        }

        response = None
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            transcription = result["choices"][0]["message"]["content"].strip()
            return transcription
        except Exception as e:
            logging.error(f"OpenRouter STT failed: {e}")
            if response is not None and hasattr(response, "text"):
                logging.error(f"Response: {response.text}")
            return ""
