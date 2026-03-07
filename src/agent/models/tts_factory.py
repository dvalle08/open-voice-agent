from __future__ import annotations

from typing import Any

from livekit.plugins import deepgram, nvidia

from src.core.logger import logger
from src.core.settings import settings
from src.plugins.pocket_tts import PocketTTS


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
        return PocketTTS(
            voice=settings.voice.POCKET_TTS_VOICE,
            temperature=settings.voice.POCKET_TTS_TEMPERATURE,
            lsd_decode_steps=settings.voice.POCKET_TTS_LSD_DECODE_STEPS,
        )

    if provider == "deepgram":
        logger.info("Initializing Deepgram TTS with plugin defaults")
        return deepgram.TTS(model="aura-2-thalia-en", api_key=settings.voice.DEEPGRAM_API_KEY)

    if provider == "nvidia":
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
