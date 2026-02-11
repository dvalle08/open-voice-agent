import asyncio

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    MetricsCollectedEvent,
    UserInputTranscribedEvent,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import langchain

from src.agent.graph import create_graph
from src.agent.metrics_collector import MetricsCollector
from src.plugins.moonshine_stt import MoonshineSTT
from src.plugins.pocket_tts import PocketTTS
from src.core.settings import settings
from src.core.logger import logger


class Assistant(Agent):
    def __init__(self, metrics_collector: MetricsCollector) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )
        self._metrics_collector = metrics_collector

    async def on_enter(self) -> None:
        """Called when the agent enters the session. Set up metrics listeners."""
        def metrics_wrapper(event: MetricsCollectedEvent) -> None:
            asyncio.create_task(
                self._metrics_collector.on_metrics_collected(event.metrics)
            )

        def transcript_wrapper(event: UserInputTranscribedEvent) -> None:
            asyncio.create_task(
                self._metrics_collector.on_user_input_transcribed(
                    event.transcript,
                    is_final=event.is_final,
                )
            )

        def conversation_item_wrapper(event: ConversationItemAddedEvent) -> None:
            item = event.item
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            asyncio.create_task(
                self._metrics_collector.on_conversation_item_added(
                    role=role,
                    content=content,
                )
            )

        self.session.on("metrics_collected", metrics_wrapper)
        self.session.on("user_input_transcribed", transcript_wrapper)
        self.session.on("conversation_item_added", conversation_item_wrapper)


server = AgentServer(num_idle_processes=settings.livekit.LIVEKIT_NUM_IDLE_PROCESSES)


@server.rtc_session(agent_name=settings.livekit.LIVEKIT_AGENT_NAME)
async def session_handler(ctx: agents.JobContext) -> None:
    # Create metrics collector
    metrics_collector = MetricsCollector(
        room=ctx.room,
        model_name=settings.voice.MOONSHINE_MODEL_ID,
    )

    def tts_metrics_callback(
        *,
        ttfb: float,
        duration: float,
        audio_duration: float,
    ) -> None:
        asyncio.create_task(
            metrics_collector.on_tts_synthesized(
                ttfb=ttfb,
                duration=duration,
                audio_duration=audio_duration,
            )
        )

    tts_engine = PocketTTS(
        voice=settings.voice.POCKET_TTS_VOICE,
        temperature=settings.voice.POCKET_TTS_TEMPERATURE,
        lsd_decode_steps=settings.voice.POCKET_TTS_LSD_DECODE_STEPS,
        metrics_callback=tts_metrics_callback,
    )

    session = AgentSession(
        stt=MoonshineSTT(model_id=settings.voice.MOONSHINE_MODEL_ID),
        llm=langchain.LLMAdapter(create_graph()),
        tts=tts_engine,
        vad=silero.VAD.load(
            min_speech_duration=settings.voice.VAD_MIN_SPEECH_DURATION,
            min_silence_duration=settings.voice.VAD_MIN_SILENCE_DURATION,
            activation_threshold=settings.voice.VAD_THRESHOLD,
        ),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(metrics_collector=metrics_collector),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                sample_rate=settings.voice.LIVEKIT_SAMPLE_RATE,
                num_channels=settings.voice.LIVEKIT_NUM_CHANNELS,
                frame_size_ms=settings.voice.LIVEKIT_FRAME_SIZE_MS,
                pre_connect_audio=settings.voice.LIVEKIT_PRE_CONNECT_AUDIO,
                pre_connect_audio_timeout=settings.voice.LIVEKIT_PRE_CONNECT_TIMEOUT,
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )
    await session.generate_reply(instructions="Greet the user and offer your assistance.")


if __name__ == "__main__":
    agents.cli.run_app(server)
