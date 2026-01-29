import pytest
from unittest.mock import AsyncMock

from src.services.audio_service import AudioService
from src.services.transcription_service import TranscriptionService
from src.services.tts_service import TTSService
from src.core.exceptions import TranscriptionError, TTSError


def test_audio_service_add_chunk():
    service = AudioService()
    
    service.add_chunk(b"chunk1")
    service.add_chunk(b"chunk2")
    
    assert service.get_buffer_size() == 2


def test_audio_service_clear_buffer():
    service = AudioService()
    
    service.add_chunk(b"chunk1")
    service.clear_buffer()
    
    assert service.get_buffer_size() == 0


def test_audio_service_get_and_clear():
    service = AudioService()
    
    service.add_chunk(b"chunk1")
    service.add_chunk(b"chunk2")
    
    buffer = service.get_and_clear_buffer()
    
    assert len(buffer) == 2
    assert service.get_buffer_size() == 0


@pytest.mark.asyncio
async def test_transcription_service_empty_audio(mock_voice_provider):
    service = TranscriptionService(mock_voice_provider)
    
    with pytest.raises(TranscriptionError):
        async for _ in service.transcribe_audio([]):
            pass


@pytest.mark.asyncio
async def test_tts_service_empty_text(mock_voice_provider):
    service = TTSService(mock_voice_provider)
    
    with pytest.raises(TTSError):
        async for _ in service.generate_speech(""):
            pass


@pytest.mark.asyncio
async def test_tts_service_whitespace_text(mock_voice_provider):
    service = TTSService(mock_voice_provider)
    
    with pytest.raises(TTSError):
        async for _ in service.generate_speech("   "):
            pass
