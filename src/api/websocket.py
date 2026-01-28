"""WebSocket connection handler for voice conversations."""

import asyncio
import base64
import json
from typing import Optional

from fastapi import WebSocket

from src.agent.graph import create_conversation_graph
from src.agent.state import ConversationState
from src.core.logger import logger
from src.core.settings import settings
from src.models.voice import GradiumConfig, GradiumProvider


class VoiceWebSocketHandler:
    """Handler for WebSocket voice conversations.
    
    Manages bidirectional audio streaming, transcription, and response generation.
    """

    def __init__(self, websocket: WebSocket):
        """Initialize WebSocket handler.
        
        Args:
            websocket: WebSocket connection
        """
        self.websocket = websocket
        self.voice_provider: Optional[GradiumProvider] = None
        self.conversation_graph = None
        self.conversation_state: ConversationState = {
            "messages": [],
            "current_transcript": "",
            "context": {},
            "turn_active": False,
        }
        self.session_id = "default"  # TODO: Generate unique session IDs

    async def connect(self):
        """Accept WebSocket connection and initialize voice provider."""
        await self.websocket.accept()
        logger.info("WebSocket connection accepted")
        
        # Initialize voice provider
        config = GradiumConfig(
            api_key=settings.voice.GRADIUM_API_KEY,
            voice_id=settings.voice.GRADIUM_VOICE_ID,
            model_name=settings.voice.GRADIUM_MODEL_NAME,
            region=settings.voice.GRADIUM_REGION,
            sample_rate_input=settings.voice.SAMPLE_RATE_INPUT,
            sample_rate_output=settings.voice.SAMPLE_RATE_OUTPUT,
            vad_threshold=settings.voice.VAD_THRESHOLD,
        )
        
        self.voice_provider = GradiumProvider(config)
        await self.voice_provider.connect()
        
        # Initialize conversation graph
        self.conversation_graph = create_conversation_graph()
        
        logger.info("Voice provider and conversation graph initialized")

    async def disconnect(self):
        """Close connections and cleanup resources."""
        if self.voice_provider:
            await self.voice_provider.disconnect()
        logger.info("WebSocket connection closed")

    async def send_json(self, data: dict):
        """Send JSON message to client.
        
        Args:
            data: Dictionary to send as JSON
        """
        try:
            await self.websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending JSON: {e}")

    async def send_error(self, message: str):
        """Send error message to client.
        
        Args:
            message: Error message
        """
        await self.send_json({"type": "error", "message": message})

    async def handle_conversation(self):
        """Main conversation loop handling audio I/O and agent responses."""
        try:
            # Start audio input and output tasks concurrently
            input_task = asyncio.create_task(self.handle_audio_input())
            
            # Wait for tasks to complete
            await input_task
            
        except Exception as e:
            logger.error(f"Error in conversation handler: {e}", exc_info=True)
            await self.send_error(str(e))

    async def handle_audio_input(self):
        """Handle incoming audio from client and process transcriptions."""
        audio_buffer = []
        
        try:
            while True:
                # Receive message from client
                data = await self.websocket.receive_json()
                msg_type = data.get("type")
                
                if msg_type == "audio":
                    # Decode audio data
                    audio_data = base64.b64decode(data.get("data", ""))
                    audio_buffer.append(audio_data)
                    
                    # Mark turn as active
                    self.conversation_state["turn_active"] = True
                    
                elif msg_type == "end_turn":
                    # User finished speaking
                    logger.info("Turn ended by client")
                    self.conversation_state["turn_active"] = False
                    
                    if audio_buffer:
                        # Process accumulated audio
                        await self.process_audio_buffer(audio_buffer)
                        audio_buffer = []
                
                elif msg_type == "start_conversation":
                    # Start new conversation
                    logger.info("Starting new conversation")
                    self.conversation_state = {
                        "messages": [],
                        "current_transcript": "",
                        "context": {},
                        "turn_active": False,
                    }
                    await self.send_json({"type": "ready"})
                
                else:
                    logger.warning(f"Unknown message type: {msg_type}")
                    
        except Exception as e:
            logger.error(f"Error handling audio input: {e}", exc_info=True)
            raise

    async def process_audio_buffer(self, audio_buffer: list[bytes]):
        """Process buffered audio through STT and generate response.
        
        Args:
            audio_buffer: List of audio chunks
        """
        try:
            # Convert audio buffer to async generator
            async def audio_generator():
                for chunk in audio_buffer:
                    yield chunk
            
            # Transcribe audio
            full_transcript = ""
            async for result in self.voice_provider.speech_to_text(audio_generator()):
                full_transcript += " " + result.text
                
                # Send transcript update to client
                await self.send_json({
                    "type": "transcript",
                    "text": result.text,
                    "start_s": result.start_s,
                })
                
                # Check VAD info
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
            
            # Update conversation state
            self.conversation_state["current_transcript"] = full_transcript
            
            # Generate response using conversation graph
            result = self.conversation_graph.invoke(
                self.conversation_state,
                config={"configurable": {"thread_id": self.session_id}},
            )
            
            # Update state with result
            self.conversation_state = result
            
            # Get AI response
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    response_text = last_message.content
                    logger.info(f"AI response: {response_text}")
                    
                    # Send text response to client
                    await self.send_json({
                        "type": "response_text",
                        "text": response_text,
                    })
                    
                    # Generate and stream audio response
                    await self.generate_audio_response(response_text)
            
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}", exc_info=True)
            await self.send_error(f"Error processing audio: {str(e)}")

    async def generate_audio_response(self, text: str):
        """Generate and stream audio response to client.
        
        Args:
            text: Text to convert to speech
        """
        try:
            # Generate TTS audio
            async for audio_chunk in self.voice_provider.text_to_speech(text):
                # Encode and send audio chunk
                encoded_audio = base64.b64encode(audio_chunk).decode("utf-8")
                await self.send_json({
                    "type": "audio",
                    "data": encoded_audio,
                })
            
            # Signal response complete
            await self.send_json({"type": "response_complete"})
            logger.info("Audio response generation complete")
            
        except Exception as e:
            logger.error(f"Error generating audio response: {e}", exc_info=True)
            await self.send_error(f"Error generating audio: {str(e)}")
