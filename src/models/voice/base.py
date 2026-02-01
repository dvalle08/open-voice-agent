from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from pydantic import BaseModel

from src.models.voice.types import TranscriptionResult, VADInfo


class VoiceProviderConfig(BaseModel):
    provider_name: str
    sample_rate_input: int = 24000
    sample_rate_output: int = 48000
    chunk_duration_ms: int = 80


class BaseVoiceProvider(ABC):
    def __init__(self, config: VoiceProviderConfig):
        self.config = config
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def text_to_speech(
        self, text: str, stream: bool = True
    ) -> AsyncIterator[bytes]:
        pass

    async def speech_to_text(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        raise NotImplementedError("Speech-to-text not supported by this provider")

    @abstractmethod
    async def get_vad_info(self) -> Optional[VADInfo]:
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
