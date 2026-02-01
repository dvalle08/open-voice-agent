import asyncio
from typing import AsyncIterator, Optional

import httpx

from src.core.logger import logger
from src.models.voice.base import BaseVoiceProvider, VoiceProviderConfig
from src.models.voice.types import TranscriptionResult, VADInfo


class NvidiaConfig(VoiceProviderConfig):
    provider_name: str = "nvidia"
    api_key: str
    language: str = "en-US"
    voice_name: str = "Magpie-Multilingual.EN-US.Aria"
    tts_model: str = "magpie-tts-multilingual"
    tts_endpoint: str = ""
    sample_rate_output: int = 48000


class NvidiaVoiceProvider(BaseVoiceProvider):
    def __init__(self, config: NvidiaConfig):
        super().__init__(config)
        self.config: NvidiaConfig = config
        self._current_vad: Optional[VADInfo] = None

    async def connect(self) -> None:
        # No connection needed for HTTP API
        self._connected = True
        logger.info("NVIDIA API TTS provider ready")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("NVIDIA API TTS provider disconnected")

    async def text_to_speech(
        self, text: str, stream: bool = True
    ) -> AsyncIterator[bytes]:
        if not self.is_connected:
            raise RuntimeError("NVIDIA API provider not connected")

        if not self.config.tts_endpoint:
            raise RuntimeError(
                "TTS requires NVIDIA_TTS_ENDPOINT to be set. "
                "Get a TTS endpoint from: https://build.nvidia.com/"
            )

        async for chunk in self._text_to_speech_http(text, stream):
            yield chunk

    async def _text_to_speech_http(
        self, text: str, stream: bool = True
    ) -> AsyncIterator[bytes]:
        endpoint = self.config.tts_endpoint.rstrip("/")
        url = f"{endpoint}/v1/audio/synthesize"

        try:
            logger.debug(f"Generating speech via HTTP API for text: {text[:50]}...")

            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "language": self.config.language,
                "text": text,
                "voice": self.config.voice_name,
                "sample_rate_hz": self.config.sample_rate_output,
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                # For streaming, we need to handle the response appropriately
                # For now, return the full content
                yield response.content

            logger.debug("HTTP TTS generation complete")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in NVIDIA TTS API: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"NVIDIA TTS API error: {e.response.status_code}") from e
        except Exception as e:
            logger.error(f"Error in NVIDIA HTTP TTS API: {e}")
            raise


    async def get_vad_info(self) -> Optional[VADInfo]:
        return self._current_vad
