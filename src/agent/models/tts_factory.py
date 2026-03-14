from __future__ import annotations

import asyncio
from typing import Any

from livekit.plugins import deepgram, nvidia
from livekit.plugins.nvidia import tts as nvidia_tts_module

from src.core.logger import logger
from src.core.settings import settings
from src.plugins.pocket_tts import PocketTTS


_NVIDIA_TTS_PATCH_SENTINEL = "_open_voice_agent_nvidia_tts_patch_applied"


def _patch_nvidia_tts_stream_once() -> None:
    """Patch the upstream NVIDIA stream until LiveKit fixes metrics/shutdown handling."""
    stream_cls = nvidia_tts_module.SynthesizeStream
    if getattr(stream_cls, _NVIDIA_TTS_PATCH_SENTINEL, False):
        return

    async def _patched_run(self: Any, output_emitter: Any) -> None:
        output_emitter.initialize(
            request_id=self._context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )
        output_emitter.start_segment(segment_id=self._context_id)

        done_fut: asyncio.Future[None] = asyncio.Future()
        synthesis_started = False

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue
                self._sent_tokenizer_stream.push_text(data)
            self._sent_tokenizer_stream.end_input()

        async def _process_segments() -> None:
            async for word_stream in self._sent_tokenizer_stream:
                self._token_q.put(word_stream)
            self._token_q.put(None)

        def _mark_started_once() -> None:
            nonlocal synthesis_started
            if synthesis_started:
                return
            synthesis_started = True
            self._mark_started()

        def _resolve_done() -> None:
            if done_fut.done():
                return
            try:
                done_fut.set_result(None)
            except asyncio.InvalidStateError:
                return

        def _synthesize_worker() -> None:
            try:
                service = self._tts._ensure_session()
                while True:
                    token = self._token_q.get()

                    if not token:
                        break

                    try:
                        self._event_loop.call_soon_threadsafe(_mark_started_once)
                        responses = service.synthesize_online(
                            token.token,
                            self._opts.voice,
                            self._opts.language_code,
                            sample_rate_hz=self._opts.sample_rate,
                            encoding=nvidia_tts_module.AudioEncoding.LINEAR_PCM,
                        )
                        for response in responses:
                            self._event_loop.call_soon_threadsafe(
                                output_emitter.push, response.audio
                            )

                    except Exception as e:
                        nvidia_tts_module.logger.error(f"Error in synthesis: {e}")
                        continue
            finally:
                try:
                    self._event_loop.call_soon_threadsafe(_resolve_done)
                except RuntimeError:
                    return

        synthesize_thread = nvidia_tts_module.threading.Thread(
            target=_synthesize_worker,
            name="nvidia-tts-synthesize",
            daemon=True,
        )
        synthesize_thread.start()

        tasks = [
            asyncio.create_task(_input_task()),
            asyncio.create_task(_process_segments()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            self._token_q.put(None)
            await done_fut
            output_emitter.end_segment()

    stream_cls._run = _patched_run
    setattr(stream_cls, _NVIDIA_TTS_PATCH_SENTINEL, True)


def create_tts() -> Any:
    """Create a TTS instance based on the configured provider."""
    provider = settings.voice.TTS_PROVIDER.lower()

    if provider == "pocket":
        logger.info(
            "Initializing Pocket TTS: voice=%s temperature=%s lsd_decode_steps=%s",
            settings.voice.POCKET_TTS_VOICE,
            settings.voice.POCKET_TTS_TEMPERATURE,
            settings.voice.POCKET_TTS_LSD_DECODE_STEPS,
        )
        pocket_kwargs: dict[str, Any] = dict(
            voice=settings.voice.POCKET_TTS_VOICE,
            temperature=settings.voice.POCKET_TTS_TEMPERATURE,
            lsd_decode_steps=settings.voice.POCKET_TTS_LSD_DECODE_STEPS,
        )
        return PocketTTS(**pocket_kwargs)

    if provider == "deepgram":
        logger.info("Initializing Deepgram TTS with plugin defaults")
        return deepgram.TTS(model="aura-2-thalia-en", api_key=settings.voice.DEEPGRAM_API_KEY)

    if provider == "nvidia":
        _patch_nvidia_tts_stream_once()
        nvidia_tts_api_key = (settings.voice.NVIDIA_TTS_API_KEY or "").strip() or None
        shared_nvidia_api_key = (settings.llm.NVIDIA_API_KEY or "").strip() or None

        if nvidia_tts_api_key:
            api_key = nvidia_tts_api_key
            key_source = "NVIDIA_TTS_API_KEY"
        elif shared_nvidia_api_key:
            api_key = shared_nvidia_api_key
            key_source = "NVIDIA_API_KEY"
        else:
            api_key = None
            key_source = "not_set"

        logger.info(
            "Initializing NVIDIA TTS: voice=%s language=%s server=%s use_ssl=%s",
            settings.voice.NVIDIA_TTS_VOICE,
            settings.voice.NVIDIA_TTS_LANGUAGE_CODE,
            settings.voice.NVIDIA_TTS_SERVER,
            settings.voice.NVIDIA_TTS_USE_SSL,
        )
        logger.info("NVIDIA TTS auth source: %s", key_source)
        return nvidia.TTS(
            voice=settings.voice.NVIDIA_TTS_VOICE,
            language_code=settings.voice.NVIDIA_TTS_LANGUAGE_CODE,
            server=settings.voice.NVIDIA_TTS_SERVER,
            function_id=settings.voice.NVIDIA_TTS_FUNCTION_ID,
            use_ssl=settings.voice.NVIDIA_TTS_USE_SSL,
            api_key=api_key,
        )

    raise ValueError(
        f"Unknown TTS provider: {provider}. Must be 'pocket', 'deepgram', or 'nvidia'"
    )
