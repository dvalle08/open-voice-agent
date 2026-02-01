"""Streamlit UI for voice agent with audio I/O."""

import asyncio
import base64
import json
from threading import Thread
from queue import Queue
from typing import Optional

import streamlit as st
import websockets

from src.core.logger import logger

# Page configuration
st.set_page_config(
    page_title="Open Voice Agent",
    page_icon="🎙️",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False
if "current_transcript" not in st.session_state:
    st.session_state.current_transcript = ""
if "processing" not in st.session_state:
    st.session_state.processing = False
if "response_queue" not in st.session_state:
    st.session_state.response_queue = Queue()


def send_audio_to_websocket(ws_url: str, audio_data: str, response_queue: Queue):
    """Send audio to WebSocket and receive responses in background thread.
    
    Args:
        ws_url: WebSocket URL
        audio_data: Base64 encoded audio data
        response_queue: Queue to put responses
    """
    async def communicate():
        try:
            async with websockets.connect(ws_url) as websocket:
                # Send audio message
                await websocket.send(json.dumps({
                    "type": "audio",
                    "data": audio_data
                }))
                logger.info("Audio sent to WebSocket")
                
                # Send end turn signal
                await websocket.send(json.dumps({"type": "end_turn"}))
                logger.info("End turn signal sent")
                
                # Receive responses
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        response = json.loads(message)
                        response_queue.put(response)
                        
                        # Stop if response is complete
                        if response.get("type") == "response_complete":
                            logger.info("Response complete received")
                            break
                            
                    except asyncio.TimeoutError:
                        logger.warning("WebSocket receive timeout")
                        break
                    except Exception as e:
                        logger.error(f"Error receiving message: {e}")
                        response_queue.put({"type": "error", "message": str(e)})
                        break
                        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            response_queue.put({"type": "error", "message": str(e)})
    
    # Run async function
    asyncio.run(communicate())


# Title and description
st.title("🎙️ Open Voice Agent")
st.markdown(
    """
    Voice conversation with AI using NVIDIA API for speech synthesis 
    and LangGraph for conversation management.
    """
)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # WebSocket connection settings
    ws_url = st.text_input(
        "WebSocket URL",
        value="ws://localhost:8000/ws/voice",
        help="URL of the FastAPI WebSocket server",
    )
    
    # Connection status
    status_color = "🟢" if st.session_state.ws_connected else "🔴"
    st.markdown(f"**Status:** {status_color} {'Connected' if st.session_state.ws_connected else 'Disconnected'}")
    st.markdown("**Voice Provider:** NVIDIA API (configured on the server)")
    
    # Test connection button
    if st.button("🔍 Test Connection"):
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ Server is running!")
                st.session_state.ws_connected = True
            else:
                st.error("❌ Server not responding correctly")
                st.session_state.ws_connected = False
        except Exception as e:
            st.error(f"❌ Cannot connect to server: {e}")
            st.session_state.ws_connected = False
    
    st.divider()
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.current_transcript = ""
        st.rerun()
    
    # Download conversation
    if st.session_state.messages:
        transcript = "\n\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages]
        )
        st.download_button(
            label="📥 Download Transcript",
            data=transcript,
            file_name="conversation_transcript.txt",
            mime="text/plain",
        )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Conversation")
    
    # Chat container
    chat_container = st.container(height=500)
    
    with chat_container:
        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Display current transcript being captured
        if st.session_state.current_transcript:
            with st.chat_message("user"):
                st.write(f"*{st.session_state.current_transcript}...*")

with col2:
    st.subheader("🎤 Audio Controls")
    
    # Check if server is running
    if not st.session_state.ws_connected:
        st.warning("⚠️ Please test connection first")
    
    # Audio recorder using built-in st.audio_input
    st.markdown("**Record your message:**")
    
    audio_data = st.audio_input(
        "Click to start/stop recording",
        key="audio_input",
        help="Click to start recording, click again to stop"
    )
    
    if audio_data is not None:
        st.success("✓ Audio recorded!")
        
        # Show audio player
        st.audio(audio_data)
        
        # Send button
        if st.button("📤 Send Audio", disabled=st.session_state.processing):
            if not st.session_state.ws_connected:
                st.error("❌ Not connected to server")
            else:
                st.session_state.processing = True
                
                # Get audio bytes
                audio_bytes = audio_data.getvalue()
                
                # Encode to base64
                encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Show processing indicator
                with st.spinner("🔊 Processing audio..."):
                    # Start WebSocket communication in background thread
                    thread = Thread(
                        target=send_audio_to_websocket,
                        args=(ws_url, encoded_audio, st.session_state.response_queue),
                        daemon=True
                    )
                    thread.start()
                    
                    # Wait for thread to complete (with timeout)
                    thread.join(timeout=30)
                    
                    # Process responses from queue
                    transcript_text = ""
                    response_text = ""
                    audio_chunks = []
                    
                    while not st.session_state.response_queue.empty():
                        response = st.session_state.response_queue.get()
                        msg_type = response.get("type")
                        
                        if msg_type == "transcript":
                            transcript_text += " " + response.get("text", "")
                        elif msg_type == "response_text":
                            response_text = response.get("text", "")
                        elif msg_type == "audio":
                            audio_chunks.append(response.get("data", ""))
                        elif msg_type == "error":
                            st.error(f"Error: {response.get('message')}")
                    
                    # Add messages to conversation
                    if transcript_text.strip():
                        st.session_state.messages.append({
                            "role": "user",
                            "content": transcript_text.strip()
                        })
                    
                    if response_text:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                
                st.session_state.processing = False
                st.success("✅ Processing complete!")
                st.rerun()
    
    # Processing indicator
    if st.session_state.processing:
        st.info("⏳ Processing your message...")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown(
            """
            1. **Test connection** first to ensure server is running
            2. **Click** the audio input to start recording
            3. **Speak** your message clearly
            4. **Click again** to stop recording
            5. **Review** the recorded audio (optional)
            6. **Send** the audio for processing
            7. Wait for the AI response (text will appear in chat)
            
            **Tips:**
            - Speak clearly and at a normal pace
            - Wait for the response before recording again
            - Keep messages concise for better results
            - Use headphones to avoid echo
            """
        )
    
    # System info
    with st.expander("ℹ️ System Info"):
        st.markdown(f"""
        **Voice Provider:** NVIDIA API  
        **WebSocket:** `{ws_url}`  
        **Messages:** {len(st.session_state.messages)}
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <small>Powered by NVIDIA API (Voice) + LangGraph (Conversations) + Streamlit (UI)</small>
    </div>
    """,
    unsafe_allow_html=True,
)
