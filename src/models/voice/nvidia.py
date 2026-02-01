import asyncio
from typing import AsyncIterator, Optional

import httpx
import riva.client

from src.core.logger import logger
from src.models.voice.base import BaseVoiceProvider, VoiceProviderConfig
from src.models.voice.types import TranscriptionResult, VADInfo


class NvidiaConfig(VoiceProviderConfig):
    provider_name: str = "nvidia"
    api_key: str
    language: str = "en-US"
    voice_name: str = "Magpie-Multilingual.EN-US.Aria"
    asr_model: str = "parakeet-ctc-0.6b-asr"
    tts_model: str = "magpie-tts-multilingual"
    grpc_server: str = "grpc.nvcf.nvidia.com:443"
    asr_function_id: str = "d8dd4e9b-fbf5-4fb0-9dba-8cf436c8d965"
    tts_endpoint: str = ""
    tts_api_type: str = "http"
    sample_rate_input: int = 24000
    sample_rate_output: int = 48000


class NvidiaVoiceProvider(BaseVoiceProvider):
    def __init__(self, config: NvidiaConfig):
        super().__init__(config)
        self.config: NvidiaConfig = config
        self.auth: Optional[riva.client.Auth] = None
        self.asr_service: Optional[riva.client.ASRService] = None
        self.tts_service: Optional[riva.client.TTSService] = None
        self._current_vad: Optional[VADInfo] = None

    async def connect(self) -> None:
        try:
            logger.info(f"Connecting to NVIDIA Riva at {self.config.grpc_server}...")
            
            self.auth = riva.client.Auth(
                ssl_cert=None,
                use_ssl=True,
                uri=self.config.grpc_server,
                metadata_args=[
                    ["function-id", self.config.asr_function_id],
                    ["authorization", f"Bearer {self.config.api_key}"]
                ]
            )
            
            self.asr_service = riva.client.ASRService(self.auth)
            
            # TTS may not be available in hosted API
            try:
                if hasattr(riva.client, 'TTSService'):
                    self.tts_service = riva.client.TTSService(self.auth)
                else:
                    logger.warning("TTSService not available in this Riva client version")
                    self.tts_service = None
            except Exception as tts_error:
                logger.warning(f"Could not initialize TTS service: {tts_error}")
                self.tts_service = None
            
            self._connected = True
            logger.info("Successfully connected to NVIDIA Riva (ASR available)")
        except Exception as e:
            logger.error(f"Failed to connect to NVIDIA Riva: {e}")
            raise

    async def disconnect(self) -> None:
        try:
            self.asr_service = None
            self.tts_service = None
            self.auth = None
            self._connected = False
            logger.info("Disconnected from NVIDIA Riva")
        except Exception as e:
            logger.error(f"Error during NVIDIA Riva disconnect: {e}")
            raise

    async def text_to_speech(
        self, text: str, stream: bool = True
    ) -> AsyncIterator[bytes]:
        if not self.is_connected:
            raise RuntimeError("NVIDIA Riva provider not connected")

        if self.config.tts_endpoint:
            async for chunk in self._text_to_speech_http(text, stream):
                yield chunk
        elif self.tts_service:
            async for chunk in self._text_to_speech_grpc(text, stream):
                yield chunk
        else:
            raise RuntimeError(
                "TTS requires a self-hosted Riva TTS NIM. "
                "Set NVIDIA_TTS_ENDPOINT (e.g., 'http://localhost:9000'). "
                "See: https://docs.nvidia.com/nim/riva/tts/latest/getting-started.html"
            )

    async def _text_to_speech_http(
        self, text: str, stream: bool = True
    ) -> AsyncIterator[bytes]:
        endpoint = self.config.tts_endpoint.rstrip("/")
        url = f"{endpoint}/v1/audio/synthesize_online" if stream else f"{endpoint}/v1/audio/synthesize"

        try:
            logger.debug(f"Generating speech via HTTP for text: {text[:50]}...")

            form_data = {
                "language": self.config.language,
                "text": text,
                "voice": self.config.voice_name,
                "sample_rate_hz": str(self.config.sample_rate_output),
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                if stream:
                    async with client.stream("POST", url, data=form_data) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk
                else:
                    response = await client.post(url, data=form_data)
                    response.raise_for_status()
                    yield response.content

            logger.debug("HTTP TTS generation complete")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in NVIDIA TTS: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"NVIDIA TTS HTTP error: {e.response.status_code}") from e
        except Exception as e:
            logger.error(f"Error in NVIDIA HTTP TTS: {e}")
            raise

    async def _text_to_speech_grpc(
        self, text: str, stream: bool = True
    ) -> AsyncIterator[bytes]:
        try:
            logger.debug(f"Generating speech via gRPC for text: {text[:50]}...")

            def _synthesize():
                return self.tts_service.synthesize_online(
                    text=text,
                    voice_name=self.config.voice_name,
                    language_code=self.config.language,
                    sample_rate_hz=self.config.sample_rate_output
                )

            responses = await asyncio.to_thread(_synthesize)

            for resp in responses:
                if resp.audio:
                    yield resp.audio

            logger.debug("gRPC TTS generation complete")

        except Exception as e:
            logger.error(f"Error in NVIDIA gRPC TTS: {e}")
            raise

    async def speech_to_text(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        if not self.is_connected or not self.asr_service:
            raise RuntimeError("NVIDIA Riva provider not connected")

        try:
            logger.debug("Starting speech-to-text transcription...")
            
            audio_chunks = []
            async for chunk in audio_stream:
                audio_chunks.append(chunk)
            
            audio_data = b"".join(audio_chunks)
            
            offline_config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self.config.language,
                sample_rate_hertz=self.config.sample_rate_input,
                max_alternatives=1,
                enable_automatic_punctuation=True,
                audio_channel_count=1,
            )
            
            def _recognize():
                return self.asr_service.offline_recognize(audio_data, offline_config)
            
            response = await asyncio.to_thread(_recognize)
            
            if response.results:
                for result in response.results:
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        transcription = TranscriptionResult(
                            text=alternative.transcript,
                            start_s=0.0,
                            is_final=True,
                            confidence=alternative.confidence if hasattr(alternative, 'confidence') else None
                        )
                        
                        yield transcription
                        logger.debug(f"Transcribed: {transcription.text}")
                        
        except Exception as e:
            logger.error(f"Error in NVIDIA Riva STT: {e}")
            raise

    async def get_vad_info(self) -> Optional[VADInfo]:
        return self._current_vad
