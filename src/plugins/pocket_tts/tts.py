# tts.py - Pocket TTS Plugin for LiveKit Agents
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

import numpy as np
import torch
from pocket_tts import TTSModel
from scipy import signal

from livekit.agents import tts
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid

logger = logging.getLogger(__name__)

# Reduce verbosity of pocket_tts library to avoid console spam
logging.getLogger("pocket_tts").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.models.tts_model").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.utils.utils").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.conditioners.text").setLevel(logging.WARNING)

type OptionalTTSMetricsCallback = Callable[..., None] | None


class PocketTTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "alba",
        temperature: float = 0.7,
        lsd_decode_steps: int = 1,
        sample_rate: int = 48000,
        metrics_callback: OptionalTTSMetricsCallback = None,
    ) -> None:
        """Initialize Pocket TTS plugin.

        Args:
            voice: Voice name (alba, marius, javert, jean, fantine, cosette, eponine, azelma)
                   or path to audio file for custom voice
            temperature: Sampling temperature (0.0-2.0)
            lsd_decode_steps: LSD decoding steps (higher = better quality, slower)
            sample_rate: Output sample rate in Hz (default 48000)
        """
        # Use the configured output sample rate (default 48000 Hz)
        self._output_sample_rate = sample_rate
        self._native_sample_rate = 24000  # Pocket TTS native rate

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=self._output_sample_rate,
            num_channels=1,
        )

        self._voice = voice
        self._temperature = temperature
        self._lsd_decode_steps = lsd_decode_steps
        self._metrics_callback = metrics_callback

        try:
            logger.info(f"Loading Pocket TTS model: temp={temperature}, lsd_steps={lsd_decode_steps}")
            self._model = TTSModel.load_model(
                temp=temperature,
                lsd_decode_steps=lsd_decode_steps,
            )
            logger.info("Pocket TTS model loaded successfully")

            logger.info(f"Loading voice state: {voice}")
            self._voice_state = self._model.get_state_for_audio_prompt(voice, truncate=True)
            logger.info(f"Voice state loaded for: {voice}")

        except FileNotFoundError as e:
            raise ValueError(f"Failed to load voice '{voice}': {e}") from e
        except Exception as e:
            logger.warning(f"Failed to load voice '{voice}': {e}, falling back to 'alba'")
            try:
                self._voice = "alba"
                self._voice_state = self._model.get_state_for_audio_prompt("alba", truncate=True)
                logger.info("Fallback to 'alba' voice successful")
            except Exception as fallback_error:
                raise ValueError(f"Failed to load Pocket TTS model: {fallback_error}") from fallback_error

    @property
    def model(self) -> str:
        return "pocket-tts"

    @property
    def provider(self) -> str:
        return "kyutai"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        """Synthesize text to speech.

        Args:
            text: Text to synthesize
            conn_options: API connection options

        Returns:
            ChunkedStream containing synthesized audio
        """
        # Use the base class helper to create ChunkedStream from SynthesizeStream
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.SynthesizeStream:
        """Create a streaming synthesis stream.

        Args:
            conn_options: API connection options

        Returns:
            PocketSynthesizeStream for progressive synthesis
        """
        return PocketSynthesizeStream(
            tts=self,
            conn_options=conn_options,
        )


class PocketSynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: PocketTTS,
        conn_options: APIConnectOptions,
    ) -> None:
        """Initialize streaming synthesis stream.

        Args:
            tts: PocketTTS instance
            conn_options: API connection options
        """
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Process input stream and generate audio progressively.

        Args:
            output_emitter: Audio emitter for pushing generated audio
        """
        request_id = shortuuid("TTS_")

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts._output_sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        text_buffer = ""

        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                if text_buffer.strip():
                    # Create a new segment for each text chunk
                    segment_id = shortuuid("SEG_")
                    output_emitter.start_segment(segment_id=segment_id)
                    await self._synthesize_segment(text_buffer, output_emitter, segment_id)
                    output_emitter.end_segment()
                    output_emitter.flush()
                    text_buffer = ""
                continue

            text_buffer += data

        # Process any remaining text
        if text_buffer.strip():
            segment_id = shortuuid("SEG_")
            output_emitter.start_segment(segment_id=segment_id)
            await self._synthesize_segment(text_buffer, output_emitter, segment_id)
            output_emitter.end_segment()
            output_emitter.flush()

    async def _synthesize_segment(
        self,
        text: str,
        output_emitter: tts.AudioEmitter,
        segment_id: str,
    ) -> None:
        """Synthesize a text segment and push audio chunks to emitter.

        Args:
            text: Text segment to synthesize
            output_emitter: Audio emitter for pushing generated audio
            segment_id: Segment ID for this synthesis
        """
        try:
            logger.info(f"Starting synthesis for segment_id={segment_id}, text: {text[:50]}...")
            start_time = time.perf_counter()

            # Generate all audio chunks in thread pool (synchronous pocket_tts call)
            def _generate() -> tuple[list[bytes], float]:
                chunks = []
                first_chunk_ttfb = -1.0
                for audio_chunk in self._tts._model.generate_audio_stream(
                    self._tts._voice_state,
                    text,
                    copy_state=True,
                ):
                    if first_chunk_ttfb < 0:
                        first_chunk_ttfb = time.perf_counter() - start_time
                    audio_bytes = self._tensor_to_pcm_bytes(audio_chunk)
                    chunks.append(audio_bytes)
                return chunks, first_chunk_ttfb

            # Run generation in background thread
            audio_chunks, first_chunk_ttfb = await asyncio.to_thread(_generate)
            generation_duration = time.perf_counter() - start_time

            logger.info(f"Generated {len(audio_chunks)} chunks for segment_id={segment_id}, pushing to emitter...")

            # Push raw PCM bytes to the emitter
            for chunk in audio_chunks:
                output_emitter.push(chunk)

            logger.info(f"Successfully pushed {len(audio_chunks)} chunks for segment_id={segment_id}")
            if self._tts._metrics_callback:
                self._tts._metrics_callback(
                    ttfb=first_chunk_ttfb,
                    duration=generation_duration,
                    audio_duration=self._bytes_to_duration(audio_chunks),
                )

        except Exception as e:
            logger.error(f"Error synthesizing segment {segment_id}: {e}", exc_info=True)
            raise

    def _tensor_to_pcm_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """Convert audio tensor to PCM bytes with resampling.

        Args:
            audio_tensor: Audio tensor with shape [samples] or [channels, samples]

        Returns:
            PCM audio bytes (int16) resampled to output sample rate
        """
        # Convert to mono if needed
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0)

        # Convert to numpy float32 array
        audio_np = audio_tensor.cpu().numpy().astype(np.float32)

        # Resample if needed (24000 Hz -> output sample rate)
        if self._tts._output_sample_rate != self._tts._native_sample_rate:
            num_samples_output = int(len(audio_np) * self._tts._output_sample_rate / self._tts._native_sample_rate)
            audio_np = signal.resample(audio_np, num_samples_output)

        # Clip to [-1, 1] and convert to int16
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767.0).astype(np.int16)

        return audio_int16.tobytes()

    def _bytes_to_duration(self, chunks: list[bytes]) -> float:
        total_bytes = sum(len(chunk) for chunk in chunks)
        samples = total_bytes / 2  # int16 mono samples
        if self._tts._output_sample_rate <= 0:
            return 0.0
        return float(samples / self._tts._output_sample_rate)
