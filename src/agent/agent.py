from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import langchain

from src.agent.graph import create_graph
from src.plugins.moonshine_stt import MoonshineSTT
from src.plugins.pocket_tts import PocketTTS
from src.core.settings import settings
from src.core.logger import logger


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
async def session_handler(ctx: agents.JobContext) -> None:
    session = AgentSession(
        stt=MoonshineSTT(model_id=settings.voice.MOONSHINE_MODEL_ID),
        llm=langchain.LLMAdapter(create_graph()),
        tts=PocketTTS(
            voice=settings.voice.POCKET_TTS_VOICE,
            temperature=settings.voice.POCKET_TTS_TEMPERATURE,
            lsd_decode_steps=settings.voice.POCKET_TTS_LSD_DECODE_STEPS,
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )
    await session.generate_reply(instructions="Greet the user and offer your assistance.")


if __name__ == "__main__":
    agents.cli.run_app(server)
