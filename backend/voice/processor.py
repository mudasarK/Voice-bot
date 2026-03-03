from backend.core.interfaces import STTProvider, TTSProvider
from backend.rag.chain import RAGChain
import logging

class VoiceProcessor:
    def __init__(self, stt_provider: STTProvider, tts_provider: TTSProvider, rag_chain: RAGChain):
        self.stt = stt_provider
        self.tts = tts_provider
        self.rag = rag_chain
        
    def process_audio(self, audio_bytes: bytes, audio_format: str = "wav") -> tuple[str, str, bytes]:
        """
        Receives audio bytes (e.g., from a user's mic).
        1. Transcribes to text
        2. Queries the RAG chain
        3. Converts response back to audio.
        Returns (user_text, bot_response_text, bot_audio_bytes)
        """
        logging.info("Transcribing audio...")
        user_text = self.stt.transcribe(audio_bytes, audio_format=audio_format)
        
        logging.info(f"User asking: {user_text}")
        if not user_text:
            return "", "I couldn't hear you clearly.", self.tts.synthesize("I couldn't hear you clearly.")
            
        logging.info("Querying RAG chain...")
        bot_response_text = self.rag.invoke(user_text)
        
        logging.info(f"Bot responding: {bot_response_text}")
        bot_audio_bytes = self.tts.synthesize(bot_response_text)
        
        return user_text, bot_response_text, bot_audio_bytes
