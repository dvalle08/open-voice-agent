from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AudioFormat(str, Enum):
    PCM = "pcm"
    WAV = "wav"
    OPUS = "opus"
    ULAW_8000 = "ulaw_8000"
    ALAW_8000 = "alaw_8000"
    PCM_16000 = "pcm_16000"
    PCM_24000 = "pcm_24000"


@dataclass
class VoiceMessage:
    type: str
    content: str | bytes
    timestamp: Optional[float] = None
    metadata: Optional[dict] = None


@dataclass
class VADInfo:
    inactivity_prob: float
    horizon_s: float
    step_idx: int
    total_duration_s: float

    @property
    def is_turn_complete(self, threshold: float = 0.5) -> bool:
        return self.inactivity_prob > threshold


@dataclass
class TranscriptionResult:
    text: str
    start_s: float
    stop_s: Optional[float] = None
    is_final: bool = False
    confidence: Optional[float] = None
    stream_id: Optional[int] = None
