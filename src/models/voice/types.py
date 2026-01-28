"""Common types and data structures for voice processing."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AudioFormat(str, Enum):
    """Supported audio formats."""

    PCM = "pcm"
    WAV = "wav"
    OPUS = "opus"
    ULAW_8000 = "ulaw_8000"
    ALAW_8000 = "alaw_8000"
    PCM_16000 = "pcm_16000"
    PCM_24000 = "pcm_24000"


@dataclass
class VoiceMessage:
    """Container for voice-related messages (text or audio)."""

    type: str  # "text", "audio", "transcript", etc.
    content: str | bytes
    timestamp: Optional[float] = None
    metadata: Optional[dict] = None


@dataclass
class VADInfo:
    """Voice Activity Detection information."""

    inactivity_prob: float
    horizon_s: float
    step_idx: int
    total_duration_s: float

    @property
    def is_turn_complete(self, threshold: float = 0.5) -> bool:
        """Check if the turn is likely complete based on inactivity probability."""
        return self.inactivity_prob > threshold


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""

    text: str
    start_s: float
    stop_s: Optional[float] = None
    is_final: bool = False
    confidence: Optional[float] = None
    stream_id: Optional[int] = None
