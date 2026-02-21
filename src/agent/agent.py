import asyncio
import base64
import json
import sys

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.agents.telemetry import set_tracer_provider
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    MetricsCollectedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import langchain
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.agent.graph import create_graph, create_stt
from src.agent.metrics_collector import MetricsCollector
from src.plugins.pocket_tts import PocketTTS
from src.core.settings import settings
from src.core.logger import logger

_langfuse_tracer_provider: TracerProvider | None = None


def _normalize_langfuse_host() -> str | None:
    host = settings.langfuse.LANGFUSE_HOST or settings.langfuse.LANGFUSE_BASE_URL
    if not host:
        return None
    return host.rstrip("/")


def _fallback_session_prefix() -> str | None:
    """Use console-prefixed fallback session id when running `... console`."""
    if any(arg == "console" for arg in sys.argv[1:]):
        return "console"
    return None


def _fallback_participant_prefix() -> str | None:
    """Use console-prefixed fallback participant id when running `... console`."""
    if any(arg == "console" for arg in sys.argv[1:]):
        return "console"
    return None


def setup_langfuse_tracer() -> TracerProvider | None:
    """Configure LiveKit telemetry tracer to export traces to Langfuse."""
    global _langfuse_tracer_provider

    if not settings.langfuse.LANGFUSE_ENABLED:
        return None
    if _langfuse_tracer_provider is not None:
        return _langfuse_tracer_provider

    host = _normalize_langfuse_host()
    public_key = settings.langfuse.LANGFUSE_PUBLIC_KEY
    secret_key = settings.langfuse.LANGFUSE_SECRET_KEY
    if not host or not public_key or not secret_key:
        logger.warning(
            "Langfuse tracing enabled but LANGFUSE_HOST/LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are missing"
        )
        return None

    try:
        auth = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("utf-8")
        span_exporter = OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}"},
        )
        tracer_provider = TracerProvider(
            resource=Resource.create(
                {
                    SERVICE_NAME: "open-voice-agent",
                    SERVICE_VERSION: getattr(agents, "__version__", "unknown"),
                    "deployment.environment": "default",
                }
            )
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        set_tracer_provider(tracer_provider)
        _langfuse_tracer_provider = tracer_provider
        logger.info("Langfuse OTEL tracing configured")
        return tracer_provider
    except Exception as exc:
        logger.warning(f"Failed to set up Langfuse tracing: {exc}")
        return None


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

        def speech_created_wrapper(event: SpeechCreatedEvent) -> None:
            asyncio.create_task(
                self._metrics_collector.on_speech_created(event.speech_handle)
            )

        def agent_state_changed_wrapper(event: AgentStateChangedEvent) -> None:
            asyncio.create_task(
                self._metrics_collector.on_agent_state_changed(
                    old_state=event.old_state,
                    new_state=event.new_state,
                )
            )

        self.session.on("metrics_collected", metrics_wrapper)
        self.session.on("user_input_transcribed", transcript_wrapper)
        self.session.on("conversation_item_added", conversation_item_wrapper)
        self.session.on("speech_created", speech_created_wrapper)
        self.session.on("agent_state_changed", agent_state_changed_wrapper)


server = AgentServer(num_idle_processes=settings.livekit.LIVEKIT_NUM_IDLE_PROCESSES)


@server.rtc_session(agent_name=settings.livekit.LIVEKIT_AGENT_NAME)
async def session_handler(ctx: agents.JobContext) -> None:
    logger.info(
        "Agent session started: room=%s job_id=%s",
        ctx.room.name,
        ctx.job.id,
    )
    trace_provider = setup_langfuse_tracer()
    if trace_provider:
        async def flush_trace(_: str) -> None:
            try:
                trace_provider.force_flush()
            except Exception as exc:
                logger.warning(f"Failed to flush Langfuse traces: {exc}")

        ctx.add_shutdown_callback(flush_trace)

    # Create metrics collector
    participant = getattr(ctx.job, "participant", None)
    initial_participant_id = getattr(participant, "identity", None)
    room_info = getattr(ctx.job, "room", None)
    initial_room_id = getattr(room_info, "sid", None) or ctx.room.name
    metrics_collector = MetricsCollector(
        room=ctx.room,
        model_name=(
            settings.stt.MOONSHINE_MODEL_ID
            if settings.stt.STT_PROVIDER == "moonshine"
            else settings.stt.NVIDIA_STT_MODEL
        ),
        room_name=ctx.room.name,
        room_id=initial_room_id,
        participant_id=initial_participant_id,
        fallback_session_prefix=_fallback_session_prefix(),
        fallback_participant_prefix=_fallback_participant_prefix(),
        langfuse_enabled=trace_provider is not None,
    )

    @ctx.room.on("data_received")
    def on_data_received(packet: rtc.DataPacket) -> None:
        if packet.topic != "session_meta":
            return
        try:
            payload = json.loads(packet.data.decode("utf-8"))
        except Exception:
            logger.debug("Ignoring invalid session_meta payload")
            return
        if payload.get("type") != "session_meta":
            return

        participant_id = payload.get("participant_id")
        if not isinstance(participant_id, str) and packet.participant:
            participant_id = packet.participant.identity
        logger.info(
            "Session metadata received from data channel: session_id=%s participant_id=%s room=%s",
            payload.get("session_id"),
            participant_id,
            ctx.room.name,
        )
        asyncio.create_task(
            metrics_collector.on_session_metadata(
                session_id=payload.get("session_id"),
                participant_id=participant_id,
            )
        )

    if isinstance(ctx.job.metadata, str) and ctx.job.metadata.strip():
        try:
            metadata = json.loads(ctx.job.metadata)
        except Exception:
            metadata = {}
        logger.info(
            "Session metadata received from dispatch: session_id=%s participant_id=%s room=%s",
            metadata.get("session_id"),
            metadata.get("participant_id"),
            ctx.room.name,
        )
        asyncio.create_task(
            metrics_collector.on_session_metadata(
                session_id=metadata.get("session_id"),
                participant_id=metadata.get("participant_id"),
            )
        )

    tts_engine = PocketTTS(
        voice=settings.voice.POCKET_TTS_VOICE,
        temperature=settings.voice.POCKET_TTS_TEMPERATURE,
        lsd_decode_steps=settings.voice.POCKET_TTS_LSD_DECODE_STEPS,
        sample_rate=settings.voice.SAMPLE_RATE_OUTPUT,
    )

    session = AgentSession(
        stt=create_stt(),
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
