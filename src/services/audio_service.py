from typing import List, AsyncIterator

from src.core.logger import logger
from src.core.exceptions import VoiceProviderError


class AudioService:
    def __init__(self):
        self._buffer: List[bytes] = []
    
    def add_chunk(self, chunk: bytes) -> None:
        self._buffer.append(chunk)
    
    def clear_buffer(self) -> None:
        self._buffer.clear()
    
    def get_buffer_size(self) -> int:
        return len(self._buffer)
    
    async def create_audio_stream(self) -> AsyncIterator[bytes]:
        for chunk in self._buffer:
            yield chunk
    
    def get_and_clear_buffer(self) -> List[bytes]:
        buffer_copy = self._buffer.copy()
        self.clear_buffer()
        return buffer_copy
