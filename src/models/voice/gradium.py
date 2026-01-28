"""Gradium voice provider implementation."""

import asyncio
import base64
import json
from typing import AsyncIterator, Optional

import gradium
from pydantic import Field

from src.core.logger import logger
from src.models.voice.base import BaseVoiceProvider, VoiceProviderConfig
from src.models.voice.types import AudioFormat, TranscriptionResult, VADInfo


class GradiumConfig(VoiceProviderConfig):
    """Configuration for Gradium voice provider."""

    provider_name: str = "gradium"
    api_key: str
    voice_id: str = "YTpq7expH9539ERJ"  # Emma voice
    model_name: str = "default"
    region: str = "eu"  # "eu" or "us"
    output_format: AudioFormat = AudioFormat.WAV
    input_format: AudioFormat = AudioFormat.PCM
    sample_rate_input: int = 24000  # Gradium STT requires 24kHz
    sample_rate_output: int = 48000  # Gradium TTS outputs 48kHz
    vad_threshold: float = 0.5
    vad_horizon_index: int = 2  # Use 2.0s horizon for turn detection


class GradiumProvider(BaseVoiceProvider):
    """Gradium implementation of the voice provider interface.
    
    Provides Speech-to-Text and Text-to-Speech using Gradium's API with:
    - Low-latency streaming for both STT and TTS
    - Voice Activity Detection for turn-taking
    - Support for multiple voices and languages
    """

    def __init__(self, config: GradiumConfig):
        """Initialize Gradium provider.
        
        Args:
            config: Gradium-specific configuration
        """
        super().__init__(config)
        self.config: GradiumConfig = config
        self.client: Optional[gradium.client.GradiumClient] = None
        self._current_vad: Optional[VADInfo] = None
        self._tts_stream = None
        self._stt_stream = None

    async def connect(self) -> None:
        """Establish connection to Gradium service."""
        try:
            logger.info(f"Connecting to Gradium ({self.config.region} region)...")
            self.client = gradium.client.GradiumClient(api_key=self.config.api_key)
            self._connected = True
            logger.info("Successfully connected to Gradium")
        except Exception as e:
            logger.error(f"Failed to connect to Gradium: {e}")
            raise

    async def disconnect(self) -> None:
        """Close connection and cleanup resources."""
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
        """Convert text to speech using Gradium TTS.
        
        Args:
            text: Text to convert to speech
            stream: Whether to stream audio chunks (always True for Gradium)
            
        Yields:
            Audio data chunks in the configured format
        """
        if not self.is_connected or not self.client:
            raise RuntimeError("Gradium provider not connected")

        try:
            logger.debug(f"Generating speech for text: {text[:50]}...")
            
            setup = {
                "model_name": self.config.model_name,
                "voice_id": self.config.voice_id,
                "output_format": self.config.output_format.value,
            }

            # Use Gradium's streaming TTS
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
        """Convert speech to text using Gradium STT.
        
        Args:
            audio_stream: Async iterator of audio data chunks (24kHz PCM)
            
        Yields:
            Transcription results with timestamps
        """
        if not self.is_connected or not self.client:
            raise RuntimeError("Gradium provider not connected")

        try:
            logger.debug("Starting speech-to-text transcription...")
            
            setup = {
                "model_name": self.config.model_name,
                "input_format": self.config.input_format.value,
            }

            # Create STT stream
            self._stt_stream = await self.client.stt_stream(
                setup=setup,
                audio_generator=audio_stream
            )

            # Process messages from STT stream
            async for message in self._stt_stream._stream:
                msg_type = message.get("type")
                
                if msg_type == "text":
                    # Text transcription result
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
                    # VAD information
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
                    # End of text segment
                    logger.debug("End of text segment")
                
                elif msg_type == "end_of_stream":
                    # End of transcription stream
                    logger.debug("STT stream ended")
                    break
                    
        except Exception as e:
            logger.error(f"Error in Gradium STT: {e}")
            raise

    async def get_vad_info(self) -> Optional[VADInfo]:
        """Get the most recent Voice Activity Detection information.
        
        Returns:
            Latest VAD info or None if not available
        """
        return self._current_vad

    def is_turn_complete(self) -> bool:
        """Check if the current turn is complete based on VAD.
        
        Returns:
            True if user has likely finished speaking
        """
        if not self._current_vad:
            return False
        return self._current_vad.inactivity_prob > self.config.vad_threshold
