# tts.py - Pocket TTS Plugin for LiveKit Agents
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import numpy as np
import torch
from pocket_tts import TTSModel
from scipy import signal

from livekit.agents import tts
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

from src.core.logger import logger
from src.core.settings import settings

# Reduce verbosity of pocket_tts library to avoid console spam
logging.getLogger("pocket_tts").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.models.tts_model").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.utils.utils").setLevel(logging.WARNING)
logging.getLogger("pocket_tts.conditioners.text").setLevel(logging.WARNING)


class PocketTTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "alba",
        temperature: float = 0.7,
        lsd_decode_steps: int = 1,
    ) -> None:
        """Initialize Pocket TTS plugin.

        Args:
            voice: Voice name (alba, marius, javert, jean, fantine, cosette, eponine, azelma)
                   or path to audio file for custom voice
            temperature: Sampling temperature (0.0-2.0)
            lsd_decode_steps: LSD decoding steps (higher = better quality, slower)
        """
        # Use the configured output sample rate (default 48000 Hz)
        self._output_sample_rate = settings.voice.SAMPLE_RATE_OUTPUT
        self._native_sample_rate = 24000  # Pocket TTS native rate

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=self._output_sample_rate,
            num_channels=1,
        )

        self._voice = voice
        self._temperature = temperature
        self._lsd_decode_steps = lsd_decode_steps

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
        # Create a stream and push the text to it
        stream = self.stream(conn_options=conn_options)
        stream.push_text(text)
        stream.flush()
        return tts.ChunkedStream(stream=stream, text=text)

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
        request_id = str(uuid.uuid4())
        segment_id = str(uuid.uuid4())

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts._output_sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=segment_id)

        text_buffer = ""

        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                if text_buffer.strip():
                    await self._synthesize_segment(text_buffer, output_emitter)
                    text_buffer = ""
                    output_emitter.flush()
                continue

            text_buffer += data

        if text_buffer.strip():
            await self._synthesize_segment(text_buffer, output_emitter)
            output_emitter.flush()

    async def _synthesize_segment(
        self,
        text: str,
        output_emitter: tts.AudioEmitter,
    ) -> None:
        """Synthesize a text segment and push audio chunks to emitter.

        Args:
            text: Text segment to synthesize
            output_emitter: Audio emitter for pushing generated audio
        """
        try:
            logger.info(f"Starting synthesis for text: {text[:50]}...")

            # Generate all audio chunks in thread pool (synchronous pocket_tts call)
            def _generate() -> list[bytes]:
                chunks = []
                for audio_chunk in self._tts._model.generate_audio_stream(
                    self._tts._voice_state,
                    text,
                    copy_state=True,
                ):
                    audio_bytes = self._tensor_to_pcm_bytes(audio_chunk)
                    chunks.append(audio_bytes)
                logger.info(f"Total chunks generated: {len(chunks)}")
                return chunks

            # Run generation in background thread
            audio_chunks = await asyncio.to_thread(_generate)

            logger.info(f"Received {len(audio_chunks)} chunks from generation, pushing to emitter...")

            # Push raw PCM bytes to the emitter
            for i, chunk in enumerate(audio_chunks):
                output_emitter.push(chunk)

            logger.info(f"Successfully pushed all {len(audio_chunks)} chunks")

        except Exception as e:
            logger.error(f"Error synthesizing segment: {e}", exc_info=True)
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
