# stt.py - Moonshine STT Plugin using ONNX Runtime
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly
import torch
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import AudioBuffer


@dataclass
class _STTOptions:
    language: str


class MoonshineSTT(stt.STT):

    def __init__(
        self,
        *,
        model_id: str = "usefulsensors/moonshine-streaming-small",
        language: str = "en",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=False)
        )

        self._model_id = model_id
        self._language = language

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self._model = MoonshineStreamingForConditionalGeneration.from_pretrained(
            self._model_id
        ).to(self._device, self._dtype)
        self._processor = AutoProcessor.from_pretrained(self._model_id)

    def _sanitize_options(self, *, language: str | None = None) -> _STTOptions:
        return _STTOptions(language=language or self._language)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        config = self._sanitize_options(language=language)

        audio_np, sample_rate = _merge_frames(buffer)
        audio_np = _ensure_mono(audio_np)
        audio_np = _resample_audio(audio_np, sample_rate, target_sample_rate=16000)

        inputs = self._processor(
            audio_np,
            return_tensors="pt",
            sampling_rate=16000,
        ).to(self._device, self._dtype)

        max_length = _max_length_from_inputs(inputs.attention_mask)
        generated_ids = self._model.generate(**inputs, max_length=max_length)
        transcription = self._processor.decode(
            generated_ids[0], skip_special_tokens=True
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
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechStream:
        config = self._sanitize_options(language=language)
        return MoonshineSTTStream(
            stt=self,
            model=self._model,
            processor=self._processor,
            device=self._device,
            dtype=self._dtype,
            language=config.language,
            conn_options=conn_options,
        )


class MoonshineSTTStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: stt.STT,
        model: MoonshineStreamingForConditionalGeneration,
        processor: AutoProcessor,
        device: torch.device,
        dtype: torch.dtype,
        language: str,
        conn_options: APIConnectOptions,
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=16000)
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype
        self._language = language

        self._buffer: list[rtc.AudioFrame] = []
        self._buffer_duration = 0.0
        self._silence_duration = 0.0

        self._silence_threshold_dbfs = -45.0
        self._silence_max_duration = 0.6
        self._segment_max_duration = 3.0

    async def _run(self) -> None:
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                await self._finalize_segment()
                continue

            frame = data
            self._buffer.append(frame)
            self._buffer_duration += frame.duration

            if _is_silent_frame(frame, threshold_dbfs=self._silence_threshold_dbfs):
                self._silence_duration += frame.duration
            else:
                self._silence_duration = 0.0

            if (
                self._buffer_duration >= self._segment_max_duration
                or self._silence_duration >= self._silence_max_duration
            ):
                await self._finalize_segment()

    async def _finalize_segment(self) -> None:
        if len(self._buffer) == 0:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language=self._language, text="")],
                )
            )
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    alternatives=[],
                    recognition_usage=stt.RecognitionUsage(audio_duration=0.0),
                )
            )
            return

        audio_np, sample_rate = _merge_frames(self._buffer)
        audio_np = _ensure_mono(audio_np)
        audio_np = _resample_audio(audio_np, sample_rate, target_sample_rate=16000)

        inputs = self._processor(
            audio_np,
            return_tensors="pt",
            sampling_rate=16000,
        ).to(self._device, self._dtype)

        max_length = _max_length_from_inputs(inputs.attention_mask)
        generated_ids = self._model.generate(**inputs, max_length=max_length)
        transcription = self._processor.decode(
            generated_ids[0], skip_special_tokens=True
        )

        audio_duration = float(len(audio_np)) / 16000 if len(audio_np) else 0.0
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        language=self._language,
                        text=transcription,
                    )
                ],
            )
        )
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(audio_duration=audio_duration),
            )
        )

        self._buffer = []
        self._buffer_duration = 0.0
        self._silence_duration = 0.0


def _merge_frames(buffer: list[rtc.AudioFrame]) -> tuple[np.ndarray, int]:
    if not buffer:
        return np.array([], dtype=np.float32), 16000

    combined = rtc.combine_audio_frames(buffer)
    sample_rate = combined.sample_rate
    samples_per_channel = combined.samples_per_channel
    num_channels = combined.num_channels

    pcm = np.frombuffer(combined.data, dtype=np.int16)
    if num_channels > 1:
        pcm = pcm.reshape(samples_per_channel, num_channels).mean(axis=1)

    audio_np = pcm.astype(np.float32) / 32768.0
    return audio_np, sample_rate


def _is_silent_frame(frame: rtc.AudioFrame, *, threshold_dbfs: float) -> bool:
    pcm = np.frombuffer(frame.data, dtype=np.int16)
    if pcm.size == 0:
        return True

    rms = np.sqrt(np.mean(pcm.astype(np.float32) ** 2))
    dbfs = 20.0 * np.log10(rms / 32768.0 + 1e-9)
    return dbfs < threshold_dbfs


def _ensure_mono(audio_np: np.ndarray) -> np.ndarray:
    if audio_np.ndim > 1:
        return np.mean(audio_np, axis=1)
    return audio_np


def _resample_audio(
    audio_np: np.ndarray,
    sample_rate: int,
    *,
    target_sample_rate: int,
) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return audio_np

    ratio_gcd = math.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // ratio_gcd
    down = sample_rate // ratio_gcd
    return resample_poly(audio_np, up=up, down=down)


def _max_length_from_inputs(attention_mask: torch.Tensor) -> int:
    token_limit_factor = 6.5 / 16000
    return int((attention_mask.sum() * token_limit_factor).max().item())
