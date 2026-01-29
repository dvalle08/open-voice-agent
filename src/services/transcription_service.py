from typing import List, AsyncIterator

from src.core.logger import logger
from src.core.exceptions import TranscriptionError
from src.models.voice.base import BaseVoiceProvider
from src.models.voice.types import TranscriptionResult


class TranscriptionService:
    def __init__(self, voice_provider: BaseVoiceProvider):
        self._voice_provider = voice_provider
    
    async def transcribe_audio(
        self, audio_chunks: List[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        if not audio_chunks:
            raise TranscriptionError("No audio chunks provided")
        
        async def audio_generator():
            for chunk in audio_chunks:
                yield chunk
        
        try:
            async for result in self._voice_provider.speech_to_text(audio_generator()):
                yield result
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}") from e
    
    async def get_full_transcript(self, audio_chunks: List[bytes]) -> str:
        transcript_parts = []
        
        async for result in self.transcribe_audio(audio_chunks):
            if result.text:
                transcript_parts.append(result.text)
        
        return " ".join(transcript_parts).strip()
