"""Abstract base classes for voice providers."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from pydantic import BaseModel

from src.models.voice.types import TranscriptionResult, VADInfo


class VoiceProviderConfig(BaseModel):
    """Base configuration for voice providers."""

    provider_name: str
    sample_rate_input: int = 24000
    sample_rate_output: int = 48000
    chunk_duration_ms: int = 80


class BaseVoiceProvider(ABC):
    """Abstract interface for voice providers.
    
    This interface allows easy swapping between different voice service providers
    (Gradium, OpenAI, ElevenLabs, Hugging Face models, etc.) without changing
    the business logic.
    """

    def __init__(self, config: VoiceProviderConfig):
        """Initialize the voice provider with configuration.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the voice service.
        
        This should handle authentication, WebSocket connections, or any
        other initialization required by the provider.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection and cleanup resources."""
        pass

    @abstractmethod
    async def text_to_speech(
        self, text: str, stream: bool = True
    ) -> AsyncIterator[bytes]:
        """Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            stream: Whether to stream audio chunks or return complete audio
            
        Yields:
            Audio data in bytes (format depends on provider configuration)
        """
        pass

    @abstractmethod
    async def speech_to_text(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        """Convert speech audio to text with streaming support.
        
        Args:
            audio_stream: Async iterator of audio data chunks
            
        Yields:
            Transcription results with timestamps and confidence scores
        """
        pass

    @abstractmethod
    async def get_vad_info(self) -> Optional[VADInfo]:
        """Get Voice Activity Detection information if available.
        
        Returns:
            VAD information or None if not supported by provider
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected and ready."""
        return self._connected

    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()
