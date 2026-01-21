"""
Audio Transcription Service
Uses Groq's Distil-Whisper model for fast speech-to-text.
"""
import logging
import io
from typing import Optional
from groq import Groq
from ..config import config

logger = logging.getLogger(__name__)

class AudioTranscriber:
    """Handles audio transcription using Groq API."""
    
    def __init__(self):
        self.client = None
        if config.GROQ_API_KEY:
            try:
                self.client = Groq(api_key=config.GROQ_API_KEY)
                logger.info("Groq Transcriber initialized")
            except Exception as e:
                logger.error(f"Failed to init Groq transcriber: {e}")
    
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Raw audio data
            filename: Virtual filename (needed for API)
            
        Returns:
            Transcribed text
        """
        if not self.client:
            logger.warning("Groq client not available for transcription")
            return ""
            
        try:
            # Create file-like object
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = filename
            
            transcription = self.client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model="distil-whisper-large-v3-en",
                response_format="json",
                language="en",  # Hint english, but model supports multi
                temperature=0.0
            )
            
            logger.info(f"Transcribed {len(audio_bytes)} bytes -> {len(transcription.text)} chars")
            return transcription.text
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return ""

# Singleton
_transcriber = None

def get_transcriber() -> AudioTranscriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = AudioTranscriber()
    return _transcriber
