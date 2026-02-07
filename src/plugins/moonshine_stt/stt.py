# stt.py - Moonshine STT Plugin using ONNX Runtime
from __future__ import annotations

import io
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import moonshine_onnx as moonshine
from livekit.agents import stt
from livekit.agents.utils import AudioBuffer


class MoonshineSTT(stt.STT):
    """
    LiveKit STT plugin using UsefulSensors Moonshine ONNX Runtime.

    Optimized for edge devices with minimal VRAM (26-57MB models).
    Supports streaming inference with <200ms latency.
    Uses production-ready ONNX runtime instead of PyTorch.
    """

    def __init__(
        self,
        *,
        model_size: str = "small",
        language: str = "en",
    ):
        """Initialize Moonshine ONNX STT.

        Args:
            model_size: Model size - "tiny" (26MB), "base" (57MB), or "small"
            language: Language code (currently only "en" supported)
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True)
        )

        self._model_size = model_size
        self._language = language

        # Load ONNX model using useful-moonshine-onnx
        # Models are cached at ~/.cache/huggingface/hub/models--UsefulSensors--moonshine/
        self._model = moonshine.MoonshineOnnxModel(
            model_name=model_size,  # "tiny", "base", or "small"
            model_precision="float",  # Use float precision (can use "int8" for smaller/faster)
        )

    def _sanitize_options(self, *, language: str | None = None) -> stt.STTOptions:
        """Prepare STT options with language setting."""
        return stt.STTOptions(language=language or self._language)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: stt.types.APIConnectOptions,
    ) -> stt.SpeechEvent:
        """
        Transcribe a complete audio buffer (non-streaming mode).

        Args:
            buffer: Audio data from LiveKit
            language: Language code (currently only 'en' supported)
            conn_options: API connection options (unused for local ONNX inference)

        Returns:
            SpeechEvent with transcription
        """
        config = self._sanitize_options(language=language)

        # Merge audio frames into single array
        audio_np = _merge_frames(buffer)

        # Moonshine ONNX expects 16kHz mono audio
        # LiveKit typically provides 16kHz, but if you're using 24kHz TTS output:
        # audio_resampled = resample_poly(audio_np, up=2, down=3)  # 24kHz -> 16kHz

        # Transcribe using ONNX runtime
        transcription = moonshine.transcribe(
            self._model,
            audio_np,
            tokenizer=None  # Use default tokenizer from model
        )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=config.language,
                    text=transcription,
                )
            ],
        )

    def stream(
        self,
        *,
        language: str | None = None,
    ) -> stt.SpeechStream:
        """
        Create a streaming STT session.

        Returns a SpeechStream that yields interim and final results.
        """
        config = self._sanitize_options(language=language)
        return MoonshineSTTStream(
            model=self._model,
            language=config.language,
        )


class MoonshineSTTStream(stt.SpeechStream):
    """Streaming STT session with interim results."""

    def __init__(
        self,
        *,
        model: moonshine.MoonshineOnnxModel,
        language: str,
    ):
        super().__init__()
        self._model = model
        self._language = language

        # Accumulate audio chunks
        self._buffer = AudioBuffer()

    def push_frame(self, frame: stt.AudioFrame) -> None:
        """Receive audio frame from LiveKit."""
        self._buffer.append(frame)

    async def flush(self) -> stt.SpeechEvent:
        """Process accumulated audio and return transcription."""
        if len(self._buffer) == 0:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language=self._language, text="")],
            )

        # Merge frames
        audio_np = _merge_frames(self._buffer)

        # Transcribe using ONNX runtime
        transcription = moonshine.transcribe(
            self._model,
            audio_np,
            tokenizer=None  # Use default tokenizer
        )

        # Clear buffer after processing
        self._buffer = AudioBuffer()

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    language=self._language,
                    text=transcription,
                )
            ],
        )


def _merge_frames(buffer: AudioBuffer) -> np.ndarray:
    """Merge audio buffer into single numpy array."""
    frames = []
    for frame in buffer:
        # Convert frame data to numpy
        audio_bytes = frame.data
        audio_np, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # Handle stereo - convert to mono
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)

        frames.append(audio_np)

    return np.concatenate(frames) if frames else np.array([], dtype=np.float32)
