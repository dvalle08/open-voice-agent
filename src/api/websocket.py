import asyncio
import base64
from typing import Optional

from fastapi import WebSocket

from src.agent.graph import create_conversation_graph
from src.agent.state import ConversationState
from src.core.logger import logger
from src.core.exceptions import TranscriptionError, TTSError, LLMError
from src.models.voice.base import BaseVoiceProvider
from src.models.voice.factory import VoiceProviderFactory
from src.services.audio_service import AudioService
from src.services.transcription_service import TranscriptionService
from src.services.conversation_service import ConversationService
from src.services.tts_service import TTSService
from src.services.session_manager import SessionManager


class VoiceWebSocketHandler:
    def __init__(
        self,
        websocket: WebSocket,
        session_manager: Optional[SessionManager] = None,
    ):
        self.websocket = websocket
        self.voice_provider: Optional[BaseVoiceProvider] = None
        self.conversation_graph = None
        self.conversation_state: ConversationState = {
            "messages": [],
            "current_transcript": "",
            "context": {},
            "turn_active": False,
        }
        
        self._session_manager = session_manager or SessionManager()
        self._session = None
        
        self._audio_service: Optional[AudioService] = None
        self._transcription_service: Optional[TranscriptionService] = None
        self._conversation_service: Optional[ConversationService] = None
        self._tts_service: Optional[TTSService] = None

    async def connect(self):
        await self.websocket.accept()
        logger.info("WebSocket connection accepted")
        
        self._session = self._session_manager.create_session()
        logger.info(f"Session created: {self._session.session_id}")
        
        self.voice_provider = VoiceProviderFactory.create_provider()
        await self.voice_provider.connect()
        
        self.conversation_graph = create_conversation_graph()
        
        self._audio_service = AudioService()
        self._transcription_service = TranscriptionService(self.voice_provider)
        self._conversation_service = ConversationService(self.conversation_graph)
        self._tts_service = TTSService(self.voice_provider)
        
        logger.info("Services initialized")

    async def disconnect(self):
        if self.voice_provider:
            await self.voice_provider.disconnect()
        
        if self._session:
            self._session_manager.delete_session(self._session.session_id)
        
        logger.info("WebSocket connection closed")

    async def send_json(self, data: dict):
        try:
            await self.websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending JSON: {e}")

    async def send_error(self, message: str):
        await self.send_json({"type": "error", "message": message})

    async def handle_conversation(self):
        try:
            input_task = asyncio.create_task(self.handle_audio_input())
            await input_task
        except Exception as e:
            logger.error(f"Error in conversation handler: {e}", exc_info=True)
            await self.send_error(str(e))

    async def handle_audio_input(self):
        try:
            while True:
                data = await self.websocket.receive_json()
                msg_type = data.get("type")
                
                if msg_type == "audio":
                    audio_data = base64.b64decode(data.get("data", ""))
                    self._audio_service.add_chunk(audio_data)
                    self.conversation_state["turn_active"] = True
                    
                elif msg_type == "end_turn":
                    logger.info("Turn ended by client")
                    self.conversation_state["turn_active"] = False
                    
                    if self._audio_service.get_buffer_size() > 0:
                        await self.process_audio_buffer()
                
                elif msg_type == "start_conversation":
                    logger.info("Starting new conversation")
                    self.conversation_state = {
                        "messages": [],
                        "current_transcript": "",
                        "context": {},
                        "turn_active": False,
                    }
                    self._audio_service.clear_buffer()
                    await self.send_json({"type": "ready"})
                
                else:
                    logger.warning(f"Unknown message type: {msg_type}")
                    
        except Exception as e:
            logger.error(f"Error handling audio input: {e}", exc_info=True)
            raise

    async def process_audio_buffer(self):
        try:
            audio_chunks = self._audio_service.get_and_clear_buffer()
            
            full_transcript = ""
            async for result in self._transcription_service.transcribe_audio(audio_chunks):
                full_transcript += " " + result.text
                
                await self.send_json({
                    "type": "transcript",
                    "text": result.text,
                    "start_s": result.start_s,
                })
                
                vad_info = await self.voice_provider.get_vad_info()
                if vad_info:
                    await self.send_json({
                        "type": "vad",
                        "inactivity_prob": vad_info.inactivity_prob,
                        "horizon_s": vad_info.horizon_s,
                    })
            
            full_transcript = full_transcript.strip()
            if not full_transcript:
                logger.warning("No transcript generated from audio")
                return
            
            logger.info(f"Full transcript: {full_transcript}")
            
            result = await self._conversation_service.process_message(
                transcript=full_transcript,
                session_id=self._session.session_id,
                current_state=self.conversation_state,
            )
            
            self.conversation_state = result
            
            response_text = self._conversation_service.get_last_response(result)
            if response_text:
                logger.info(f"AI response: {response_text}")
                
                await self.send_json({
                    "type": "response_text",
                    "text": response_text,
                })
                
                await self.generate_audio_response(response_text)
            
        except TranscriptionError as e:
            logger.error(f"Transcription error: {e}")
            await self.send_error(f"Transcription failed: {str(e)}")
        except LLMError as e:
            logger.error(f"LLM error: {e}")
            await self.send_error(f"Response generation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}", exc_info=True)
            await self.send_error(f"Error processing audio: {str(e)}")

    async def generate_audio_response(self, text: str):
        try:
            async for audio_chunk in self._tts_service.generate_speech(text):
                encoded_audio = base64.b64encode(audio_chunk).decode("utf-8")
                await self.send_json({
                    "type": "audio",
                    "data": encoded_audio,
                })
            
            await self.send_json({"type": "response_complete"})
            logger.info("Audio response generation complete")
            
        except TTSError as e:
            logger.error(f"TTS error: {e}")
            await self.send_error(f"Speech generation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating audio response: {e}", exc_info=True)
            await self.send_error(f"Error generating audio: {str(e)}")
