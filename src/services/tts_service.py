from typing import AsyncIterator

from src.core.logger import logger
from src.core.exceptions import TTSError
from src.models.voice.base import BaseVoiceProvider


class TTSService:
    def __init__(self, voice_provider: BaseVoiceProvider):
        self._voice_provider = voice_provider
    
    async def generate_speech(self, text: str) -> AsyncIterator[bytes]:
        if not text or not text.strip():
            raise TTSError("Empty text provided for TTS")
        
        try:
            async for audio_chunk in self._voice_provider.text_to_speech(text):
                yield audio_chunk
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise TTSError(f"Failed to generate speech: {str(e)}") from e
