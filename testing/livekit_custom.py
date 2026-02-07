from __future__ import annotations

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer,AgentSession, Agent, room_io
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import os
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from livekit.plugins import langchain

from livekit.agents import stt, tts
from huggingface_hub import InferenceClient
import io
import wave
from src.plugins.moonshine_stt import MoonshineSTT

load_dotenv(".env")

# Simple LangGraph workflow with NVIDIA LLM
def create_nvidia_workflow():
    """Create a simple LangGraph workflow with NVIDIA ChatNVIDIA"""
    
    # Initialize NVIDIA LLM
    nvidia_llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.7,
        max_tokens=150
    )
    
    # Define the conversation node
    def call_model(state: MessagesState):
        """Simple node that calls the NVIDIA LLM"""
        response = nvidia_llm.invoke(state["messages"])
        return {"messages": [response]}
    
    # Build the graph
    workflow = StateGraph(MessagesState)
    
    # Add the single node
    workflow.add_node("agent", call_model)
    
    # Define the flow: START -> agent -> END
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    
    # Compile and return
    return workflow.compile()
    


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt=MoonshineSTT(model_id="UsefulSensors/moonshine-streaming-medium"),
        llm=langchain.LLMAdapter(create_nvidia_workflow()),
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)