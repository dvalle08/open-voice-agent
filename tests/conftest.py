import pytest
from unittest.mock import Mock, AsyncMock

from src.agent.graph import create_conversation_graph
from src.models.voice.base import BaseVoiceProvider
from src.models.voice.types import TranscriptionResult, VADInfo
from src.services.session_manager import SessionManager


@pytest.fixture
def mock_voice_provider():
    provider = AsyncMock(spec=BaseVoiceProvider)
    provider.is_connected = True
    provider.connect = AsyncMock()
    provider.disconnect = AsyncMock()
    provider.get_vad_info = AsyncMock(return_value=None)
    return provider


@pytest.fixture
def mock_transcription_result():
    return TranscriptionResult(
        text="Hello world",
        start_s=0.0,
        stop_s=1.0,
        is_final=True,
    )


@pytest.fixture
def mock_vad_info():
    return VADInfo(
        inactivity_prob=0.8,
        horizon_s=2.0,
        step_idx=10,
        total_duration_s=5.0,
    )


@pytest.fixture
def session_manager():
    return SessionManager(session_timeout=3600)


@pytest.fixture
def conversation_graph():
    return create_conversation_graph()


@pytest.fixture
def sample_audio_bytes():
    return b"\x00\x01\x02\x03" * 1000
