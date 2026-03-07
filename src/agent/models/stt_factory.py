from __future__ import annotations

from livekit.plugins import deepgram, nvidia

from src.core.logger import logger
from src.core.settings import settings
from src.plugins.moonshine_stt import MoonshineSTT


def create_stt():
    """Create an STT instance based on the configured provider."""
    provider = settings.stt.STT_PROVIDER.lower()

    if provider == "nvidia":
        logger.info(
            f"Initializing NVIDIA STT: {settings.stt.NVIDIA_STT_MODEL} "
            f"(language: {settings.stt.NVIDIA_STT_LANGUAGE_CODE})"
        )
        if settings.stt.NVIDIA_STT_API_KEY:
            api_key = settings.stt.NVIDIA_STT_API_KEY
            key_source = "NVIDIA_STT_API_KEY"
        elif settings.llm.NVIDIA_API_KEY:
            api_key = settings.llm.NVIDIA_API_KEY
            key_source = "NVIDIA_API_KEY"
        else:
            api_key = None
            key_source = "not_set"

        logger.info("NVIDIA STT auth source: %s", key_source)
        if not api_key:
            logger.warning(
                "NVIDIA STT is configured but no API key is set (NVIDIA_STT_API_KEY/NVIDIA_API_KEY)"
            )

        return nvidia.STT(
            language_code=settings.stt.NVIDIA_STT_LANGUAGE_CODE,
            model=settings.stt.NVIDIA_STT_MODEL,
            api_key=api_key,
        )

    if provider == "deepgram":
        logger.info(
            "Initializing Deepgram STT: %s (language: %s)",
            settings.stt.DEEPGRAM_STT_MODEL,
            settings.stt.DEEPGRAM_STT_LANGUAGE,
        )
        return deepgram.STT(
            model=settings.stt.DEEPGRAM_STT_MODEL,
            language=settings.stt.DEEPGRAM_STT_LANGUAGE,
            api_key=settings.voice.DEEPGRAM_API_KEY,
        )

    if provider == "moonshine":
        logger.info(
            f"Initializing Moonshine STT: {settings.stt.MOONSHINE_MODEL_ID} "
            f"(language: {settings.stt.MOONSHINE_LANGUAGE})"
        )
        return MoonshineSTT(
            model_id=settings.stt.MOONSHINE_MODEL_ID,
            language=settings.stt.MOONSHINE_LANGUAGE,
        )

    raise ValueError(
        f"Unknown STT provider: {provider}. Must be 'moonshine', 'nvidia', or 'deepgram'"
    )
