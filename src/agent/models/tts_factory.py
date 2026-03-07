from __future__ import annotations

from typing import Any

from livekit.plugins import deepgram

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

    raise ValueError(f"Unknown TTS provider: {provider}. Must be 'pocket' or 'deepgram'")
