import asyncio
import base64
import json
import sys
from time import monotonic
from typing import Any

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, llm, room_io
from livekit.agents.types import APIConnectOptions
from livekit.agents.telemetry import set_tracer_provider
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.agent.llm_runtime import (
    MCP_STARTUP_GREETING_TIMEOUT_SEC,
    _install_mcp_generate_reply_guard,
    _run_startup_greeting,
    build_llm_runtime,
)
from src.agent.metrics_collector import MetricsCollector
from src.agent.stt_factory import create_stt
from src.plugins.pocket_tts import PocketTTS
from src.core.logger import detach_default_root_handler, logger
from src.core.settings import settings

_langfuse_tracer_provider: TracerProvider | None = None
ASSISTANT_INSTRUCTIONS = """You are Open Voice Agent, a helpful voice AI assistant.
You run as a real-time pipeline: LiveKit transports user audio, STT converts speech to text, the LLM reasons and may call MCP tools, and TTS generates spoken audio responses.
Default behavior: answer with the fewest words possible while still being correct and helpful.
Keep most responses to one short sentence. For greetings, acknowledgements, thanks, and casual small talk, reply in 2 to 5 words and do not call tools.
Call MCP tools only when user intent clearly requires external or up-to-date information, or when the user explicitly asks you to look something up.
If a request can be answered directly from context and general knowledge, do not call tools.
If tool usage is uncertain, ask one short clarification question before calling any tool.
Only call tools that are explicitly available in the current session; never invent tool or function names.
If a user asks how you work, explain the pipeline and component roles in one to two short sentences.
Use plain voice-friendly text only; no markdown, emojis, bullets, or decorative punctuation."""


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
                    "deployment.environment": settings.langfuse.LANGFUSE_ENVIRONMENT,
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


def _error_type_name(error_obj: Any) -> str:
    return getattr(error_obj, "type", type(error_obj).__name__)


def _error_recoverable(error_obj: Any) -> str:
    recoverable = getattr(error_obj, "recoverable", None)
    if recoverable is None:
        return "unknown"
    return str(bool(recoverable)).lower()


def _error_detail(error_obj: Any) -> str:
    nested_error = getattr(error_obj, "error", None)
    if nested_error:
        return str(nested_error)
    return str(error_obj)


async def _monitor_startup_greeting_handle(
    greeting_handle: Any,
    *,
    timeout_sec: float = MCP_STARTUP_GREETING_TIMEOUT_SEC,
) -> None:
    speech_id = getattr(greeting_handle, "id", None)
    wait_for_playout = getattr(greeting_handle, "wait_for_playout", None)
    interrupt = getattr(greeting_handle, "interrupt", None)

    if not callable(wait_for_playout):
        logger.warning(
            "Startup greeting handle missing wait_for_playout; speech_id=%s",
            speech_id,
        )
        return

    try:
        await asyncio.wait_for(wait_for_playout(), timeout=timeout_sec)
    except TimeoutError:
        logger.warning(
            "MCP startup greeting timed out after %.2fs; interrupting speech_id=%s",
            timeout_sec,
            speech_id,
        )
        if callable(interrupt):
            try:
                interrupt(force=True)
            except Exception as exc:
                logger.warning("Failed to interrupt timed out startup greeting: %s", exc)
    except asyncio.CancelledError:
        if callable(interrupt):
            try:
                interrupt(force=True)
            except Exception as exc:
                logger.warning("Failed to interrupt cancelled startup greeting: %s", exc)
        logger.info("MCP startup greeting monitor cancelled: speech_id=%s", speech_id)
    except Exception as exc:
        logger.warning("MCP startup greeting monitor failed: %s", exc)


def _schedule_startup_greeting_task(
    session: AgentSession,
    *,
    mcp_runtime_active: bool,
) -> asyncio.Task[Any] | None:
    greeting_handle = _run_startup_greeting(
        session,
        mcp_runtime_active=mcp_runtime_active,
    )
    if greeting_handle is None:
        return None

    speech_id = getattr(greeting_handle, "id", None)
    logger.info(
        "Scheduling startup greeting monitor task: mcp_runtime_active=%s speech_id=%s",
        mcp_runtime_active,
        speech_id,
    )
    task = asyncio.create_task(
        _monitor_startup_greeting_handle(greeting_handle),
        name="startup-greeting-monitor",
    )
    setattr(task, "_open_voice_startup_greeting_handle", greeting_handle)

    def _on_done(completed_task: asyncio.Task[Any]) -> None:
        if completed_task.cancelled():
            return
        try:
            exc = completed_task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.warning(f"Startup greeting monitor task failed: {exc}")

    task.add_done_callback(_on_done)
    return task


async def _run_llm_warmup(
    *,
    llm_client: Any,
    conn_options: APIConnectOptions,
    provider: str,
    model: str,
) -> None:
    started = monotonic()
    stream: Any | None = None
    got_first_chunk = False

    try:
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="Reply with OK.")
        stream = llm_client.chat(
            chat_ctx=chat_ctx,
            tools=None,
            conn_options=conn_options,
        )
        async for _ in stream:
            got_first_chunk = True
            break
    except asyncio.CancelledError:
        logger.info("LLM warm-up cancelled: provider=%s model=%s", provider, model)
    except Exception as exc:
        logger.warning("LLM warm-up failed: provider=%s model=%s detail=%s", provider, model, exc)
    finally:
        if stream is not None:
            aclose = getattr(stream, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:
                    pass

        elapsed_ms = max((monotonic() - started) * 1000.0, 0.0)
        logger.info(
            "LLM warm-up completed: provider=%s model=%s first_chunk=%s elapsed_ms=%.1f",
            provider,
            model,
            got_first_chunk,
            elapsed_ms,
        )


def _schedule_llm_warmup_task(
    *,
    llm_client: Any,
    conn_options: APIConnectOptions,
    provider: str,
    model: str,
) -> asyncio.Task[Any]:
    logger.info("Scheduling LLM warm-up task: provider=%s model=%s", provider, model)
    task = asyncio.create_task(
        _run_llm_warmup(
            llm_client=llm_client,
            conn_options=conn_options,
            provider=provider,
            model=model,
        ),
        name="llm-warmup",
    )

    def _on_done(completed_task: asyncio.Task[Any]) -> None:
        if completed_task.cancelled():
            return
        try:
            exc = completed_task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.warning(f"LLM warm-up task failed: {exc}")

    task.add_done_callback(_on_done)
    return task


async def _cancel_task_for_shutdown(
    task: asyncio.Task[Any] | None,
    *,
    task_name: str,
    timeout_sec: float = 0.5,
) -> None:
    if task is None or task.done():
        return

    greeting_handle = getattr(task, "_open_voice_startup_greeting_handle", None)
    if greeting_handle is not None:
        interrupt = getattr(greeting_handle, "interrupt", None)
        if callable(interrupt):
            try:
                interrupt(force=True)
            except Exception as exc:
                logger.warning("%s handle interrupt failed during shutdown: %s", task_name, exc)

    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=timeout_sec)
    except asyncio.CancelledError:
        logger.info("%s task cancelled during shutdown", task_name)
    except TimeoutError:
        logger.warning("%s task did not cancel within %.2fs", task_name, timeout_sec)
    except Exception as exc:
        logger.warning("%s task raised during shutdown cancellation: %s", task_name, exc)


class Assistant(Agent):
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        *,
        room_name: str,
        job_id: str,
    ) -> None:
        super().__init__(
            instructions=ASSISTANT_INSTRUCTIONS,
        )
        self._metrics_collector = metrics_collector
        self._room_name = room_name
        self._job_id = job_id

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

        def function_tools_executed_wrapper(event: FunctionToolsExecutedEvent) -> None:
            asyncio.create_task(
                self._metrics_collector.on_function_tools_executed(
                    function_calls=event.function_calls,
                    function_call_outputs=event.function_call_outputs,
                    created_at=event.created_at,
                )
            )

        def agent_state_changed_wrapper(event: AgentStateChangedEvent) -> None:
            asyncio.create_task(
                self._metrics_collector.on_agent_state_changed(
                    old_state=event.old_state,
                    new_state=event.new_state,
                )
            )

        def error_wrapper(event: ErrorEvent) -> None:
            source = type(event.source).__name__
            error_type = _error_type_name(event.error)
            recoverable = _error_recoverable(event.error)
            detail = _error_detail(event.error)
            logger.error(
                "Agent session pipeline error: room=%s job_id=%s source=%s error_type=%s recoverable=%s detail=%s",
                self._room_name,
                self._job_id,
                source,
                error_type,
                recoverable,
                detail,
            )

        def close_wrapper(event: CloseEvent) -> None:
            reason = event.reason.value
            if event.error is None:
                logger.info(
                    "Agent session closed: room=%s job_id=%s reason=%s",
                    self._room_name,
                    self._job_id,
                    reason,
                )
                return

            error_type = _error_type_name(event.error)
            recoverable = _error_recoverable(event.error)
            detail = _error_detail(event.error)
            logger.warning(
                "Agent session closed with error: room=%s job_id=%s reason=%s error_type=%s recoverable=%s detail=%s",
                self._room_name,
                self._job_id,
                reason,
                error_type,
                recoverable,
                detail,
            )

        self.session.on("metrics_collected", metrics_wrapper)
        self.session.on("user_input_transcribed", transcript_wrapper)
        self.session.on("conversation_item_added", conversation_item_wrapper)
        self.session.on("speech_created", speech_created_wrapper)
        self.session.on("function_tools_executed", function_tools_executed_wrapper)
        self.session.on("agent_state_changed", agent_state_changed_wrapper)
        self.session.on("error", error_wrapper)
        self.session.on("close", close_wrapper)

server = AgentServer(
    num_idle_processes=settings.livekit.LIVEKIT_NUM_IDLE_PROCESSES,
    job_memory_warn_mb=settings.livekit.LIVEKIT_JOB_MEMORY_WARN_MB,
)


@server.rtc_session(agent_name=settings.livekit.LIVEKIT_AGENT_NAME)
async def session_handler(ctx: agents.JobContext) -> None:
    logger.info(
        "Agent session started: room=%s job_id=%s",
        ctx.room.name,
        ctx.job.id,
    )
    trace_provider = setup_langfuse_tracer()
    startup_greeting_task: asyncio.Task[Any] | None = None
    llm_warmup_task: asyncio.Task[Any] | None = None

    if trace_provider:
        async def flush_trace(_: str) -> None:
            try:
                trace_provider.force_flush()
            except Exception as exc:
                logger.warning(f"Failed to flush Langfuse traces: {exc}")

        ctx.add_shutdown_callback(flush_trace)

    async def cancel_startup_greeting(_: str) -> None:
        await _cancel_task_for_shutdown(
            startup_greeting_task,
            task_name="startup greeting",
        )

    ctx.add_shutdown_callback(cancel_startup_greeting)

    async def cancel_llm_warmup(_: str) -> None:
        await _cancel_task_for_shutdown(
            llm_warmup_task,
            task_name="llm warm-up",
        )

    ctx.add_shutdown_callback(cancel_llm_warmup)

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
    )
    llm_conn_options = APIConnectOptions(
        max_retry=settings.llm.LLM_CONN_MAX_RETRY,
        retry_interval=settings.llm.LLM_CONN_RETRY_INTERVAL_SEC,
        timeout=settings.llm.LLM_CONN_TIMEOUT_SEC,
    )
    session_conn_options = SessionConnectOptions(llm_conn_options=llm_conn_options)
    llm_runtime = build_llm_runtime(
        llm_provider=settings.llm.LLM_PROVIDER,
        llm_temperature=settings.llm.LLM_TEMPERATURE,
        llm_max_tokens=settings.llm.LLM_MAX_TOKENS,
        llm_timeout_sec=settings.llm.LLM_CONN_TIMEOUT_SEC,
        nvidia_api_key=settings.llm.NVIDIA_API_KEY,
        nvidia_model=settings.llm.NVIDIA_MODEL,
        ollama_base_url=settings.llm.OLLAMA_BASE_URL,
        ollama_model=settings.llm.OLLAMA_MODEL,
        ollama_api_key=settings.llm.OLLAMA_API_KEY,
        mcp_enabled=settings.llm.MCP_ENABLED,
        mcp_server_url=settings.llm.MCP_SERVER_URL,
    )
    mcp_runtime_active = llm_runtime.mcp_runtime_active
    llm_warmup_task = _schedule_llm_warmup_task(
        llm_client=llm_runtime.llm,
        conn_options=llm_conn_options,
        provider=llm_runtime.provider,
        model=llm_runtime.model,
    )

    session_kwargs: dict[str, Any] = dict(
        stt=create_stt(),
        llm=llm_runtime.llm,
        tts=tts_engine,
        vad=silero.VAD.load(
            min_speech_duration=settings.voice.VAD_MIN_SPEECH_DURATION,
            min_silence_duration=settings.voice.VAD_MIN_SILENCE_DURATION,
            activation_threshold=settings.voice.VAD_THRESHOLD,
        ),
        turn_detection=MultilingualModel(),
        min_endpointing_delay=settings.voice.MIN_ENDPOINTING_DELAY,
        max_endpointing_delay=settings.voice.MAX_ENDPOINTING_DELAY,
        preemptive_generation=settings.voice.PREEMPTIVE_GENERATION,
        conn_options=session_conn_options,
    )
    if llm_runtime.mcp_servers is not None:
        session_kwargs["mcp_servers"] = llm_runtime.mcp_servers

    session = AgentSession(**session_kwargs)
    _install_mcp_generate_reply_guard(session, mcp_runtime_active=mcp_runtime_active)

    await session.start(
        room=ctx.room,
        record=False,
        agent=Assistant(
            metrics_collector=metrics_collector,
            room_name=ctx.room.name,
            job_id=ctx.job.id,
        ),
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
    if mcp_runtime_active:
        startup_greeting_task = _schedule_startup_greeting_task(
            session,
            mcp_runtime_active=mcp_runtime_active,
        )
    else:
        _run_startup_greeting(session, mcp_runtime_active=mcp_runtime_active)


if __name__ == "__main__":
    detach_default_root_handler()
    agents.cli.run_app(server)
