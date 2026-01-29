from typing import AsyncIterator, Optional

import gradium

from src.core.logger import logger
from src.models.voice.base import BaseVoiceProvider, VoiceProviderConfig
from src.models.voice.types import AudioFormat, TranscriptionResult, VADInfo


class GradiumConfig(VoiceProviderConfig):
    provider_name: str = "gradium"
    api_key: str
    voice_id: str = "YTpq7expH9539ERJ"
    model_name: str = "default"
    region: str = "eu"
    output_format: AudioFormat = AudioFormat.WAV
    input_format: AudioFormat = AudioFormat.PCM
    sample_rate_input: int = 24000
    sample_rate_output: int = 48000
    vad_threshold: float = 0.5
    vad_horizon_index: int = 2


class GradiumProvider(BaseVoiceProvider):
    def __init__(self, config: GradiumConfig):
        super().__init__(config)
        self.config: GradiumConfig = config
        self.client: Optional[gradium.client.GradiumClient] = None
        self._current_vad: Optional[VADInfo] = None
        self._tts_stream = None
        self._stt_stream = None

    async def connect(self) -> None:
        try:
            logger.info(f"Connecting to Gradium ({self.config.region} region)...")
            self.client = gradium.client.GradiumClient(api_key=self.config.api_key)
            self._connected = True
            logger.info("Successfully connected to Gradium")
        except Exception as e:
            logger.error(f"Failed to connect to Gradium: {e}")
            raise

    async def disconnect(self) -> None:
        try:
            if self._tts_stream:
                await self._tts_stream.close()
                self._tts_stream = None
            
            if self._stt_stream:
                await self._stt_stream.close()
                self._stt_stream = None
            
            self._connected = False
            logger.info("Disconnected from Gradium")
        except Exception as e:
            logger.error(f"Error during Gradium disconnect: {e}")
            raise

    async def text_to_speech(
        self, text: str, stream: bool = True
    ) -> AsyncIterator[bytes]:
        if not self.is_connected or not self.client:
            raise RuntimeError("Gradium provider not connected")

        try:
            logger.debug(f"Generating speech for text: {text[:50]}...")
            
            setup = {
                "model_name": self.config.model_name,
                "voice_id": self.config.voice_id,
                "output_format": self.config.output_format.value,
            }

            self._tts_stream = await self.client.tts_stream(setup=setup, text=text)
            
            async for audio_chunk in self._tts_stream.iter_bytes():
                if audio_chunk:
                    yield audio_chunk
                    
            logger.debug("TTS generation complete")
            
        except Exception as e:
            logger.error(f"Error in Gradium TTS: {e}")
            raise

    async def speech_to_text(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        if not self.is_connected or not self.client:
            raise RuntimeError("Gradium provider not connected")

        try:
            logger.debug("Starting speech-to-text transcription...")
            
            setup = {
                "model_name": self.config.model_name,
                "input_format": self.config.input_format.value,
            }

            self._stt_stream = await self.client.stt_stream(
                setup=setup,
                audio_generator=audio_stream
            )

            async for message in self._stt_stream._stream:
                msg_type = message.get("type")
                
                if msg_type == "text":
                    result = TranscriptionResult(
                        text=message.get("text", ""),
                        start_s=message.get("start_s", 0.0),
                        stop_s=message.get("stop_s"),
                        stream_id=message.get("stream_id"),
                        is_final=True,
                    )
                    yield result
                    logger.debug(f"Transcribed: {result.text}")
                
                elif msg_type == "step":
                    vad_data = message.get("vad", [])
                    if vad_data and len(vad_data) > self.config.vad_horizon_index:
                        horizon_data = vad_data[self.config.vad_horizon_index]
                        self._current_vad = VADInfo(
                            inactivity_prob=horizon_data.get("inactivity_prob", 0.0),
                            horizon_s=horizon_data.get("horizon_s", 2.0),
                            step_idx=message.get("step_idx", 0),
                            total_duration_s=message.get("total_duration_s", 0.0),
                        )
                        logger.debug(
                            f"VAD: inactivity_prob={self._current_vad.inactivity_prob:.2f}"
                        )
                
                elif msg_type == "end_text":
                    logger.debug("End of text segment")
                
                elif msg_type == "end_of_stream":
                    logger.debug("STT stream ended")
                    break
                    
        except Exception as e:
            logger.error(f"Error in Gradium STT: {e}")
            raise

    async def get_vad_info(self) -> Optional[VADInfo]:
        return self._current_vad

    def is_turn_complete(self) -> bool:
        if not self._current_vad:
            return False
        return self._current_vad.inactivity_prob > self.config.vad_threshold
