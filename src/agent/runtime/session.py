"""LiveKit server and session handler wiring."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, room_io
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from src.agent.models.llm_runtime import (
    build_llm_runtime,
    install_mcp_generate_reply_guard,
    run_startup_greeting,
)
from src.agent.models.tts_factory import create_tts
from src.agent.runtime.connect_options import build_api_connect_options
from src.agent.models.stt_factory import create_stt
from src.agent.runtime.assistant import Assistant
from src.agent.runtime.tasks import (
    cancel_task_for_shutdown,
    run_llm_warmup,
    schedule_startup_greeting_task,
)
from src.agent.tools.feedback import ToolFeedbackController
from src.agent.traces.langfuse import setup_langfuse_tracer
from src.agent.traces.metrics_collector import MetricsCollector
from src.core.logger import logger
from src.core.settings import settings


def _build_server() -> AgentServer:
    return AgentServer(
        num_idle_processes=settings.livekit.LIVEKIT_NUM_IDLE_PROCESSES,
        job_memory_warn_mb=settings.livekit.LIVEKIT_JOB_MEMORY_WARN_MB,
        initialize_process_timeout=(
            settings.livekit.LIVEKIT_INITIALIZE_PROCESS_TIMEOUT_SEC
        ),
    )


server = _build_server()


def fallback_session_prefix() -> str | None:
    """Use console-prefixed fallback session id when running `... console`."""
    if any(arg == "console" for arg in sys.argv[1:]):
        return "console"
    return None


def fallback_participant_prefix() -> str | None:
    """Use console-prefixed fallback participant id when running `... console`."""
    if any(arg == "console" for arg in sys.argv[1:]):
        return "console"
    return None


def _build_session_connect_options() -> tuple[APIConnectOptions, SessionConnectOptions]:
    llm_conn_options = build_api_connect_options(
        max_retry=settings.llm.LLM_CONN_MAX_RETRY,
        retry_interval_sec=settings.llm.LLM_CONN_RETRY_INTERVAL_SEC,
        timeout_sec=settings.llm.LLM_CONN_TIMEOUT_SEC,
    )
    tts_conn_options = build_api_connect_options(
        max_retry=settings.llm.LLM_CONN_MAX_RETRY,
        retry_interval_sec=settings.llm.LLM_CONN_RETRY_INTERVAL_SEC,
        timeout_sec=settings.voice.POCKET_TTS_CONN_TIMEOUT_SEC,
    )
    session_conn_options = SessionConnectOptions(
        llm_conn_options=llm_conn_options,
        tts_conn_options=tts_conn_options,
    )
    return llm_conn_options, session_conn_options


@server.rtc_session(agent_name=settings.livekit.LIVEKIT_AGENT_NAME)
async def session_handler(ctx: agents.JobContext) -> None:
    logger.info(
        "Agent session started: room=%s job_id=%s",
        ctx.room.name,
        ctx.job.id,
    )
    trace_provider = setup_langfuse_tracer()
    startup_greeting_task: asyncio.Task[Any] | None = None
    tool_feedback = ToolFeedbackController(enabled=False)

    if trace_provider:

        async def flush_trace(_: str) -> None:
            try:
                trace_provider.force_flush()
            except Exception as exc:
                logger.warning(f"Failed to flush Langfuse traces: {exc}")

        ctx.add_shutdown_callback(flush_trace)

    async def cancel_startup_greeting(_: str) -> None:
        await cancel_task_for_shutdown(
            startup_greeting_task,
            task_name="startup greeting",
        )

    ctx.add_shutdown_callback(cancel_startup_greeting)

    async def close_tool_feedback(_: str) -> None:
        await tool_feedback.aclose()

    ctx.add_shutdown_callback(close_tool_feedback)

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
        fallback_session_prefix=fallback_session_prefix(),
        fallback_participant_prefix=fallback_participant_prefix(),
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

    tts_engine = create_tts()
    llm_conn_options, session_conn_options = _build_session_connect_options()
    llm_runtime = build_llm_runtime(settings.llm)
    mcp_runtime_active = llm_runtime.mcp_runtime_active
    tool_feedback = ToolFeedbackController(enabled=mcp_runtime_active)
    logger.info(
        "Running LLM warm-up before session start: provider=%s model=%s",
        llm_runtime.provider,
        llm_runtime.model,
    )
    await run_llm_warmup(
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
    install_mcp_generate_reply_guard(session, mcp_runtime_active=mcp_runtime_active)

    await session.start(
        room=ctx.room,
        record=False,
        agent=Assistant(
            metrics_collector=metrics_collector,
            room_name=ctx.room.name,
            job_id=ctx.job.id,
            tool_feedback=tool_feedback,
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
    await tool_feedback.start(room=ctx.room, session=session)
    if mcp_runtime_active:
        startup_greeting_task = schedule_startup_greeting_task(
            session,
            mcp_runtime_active=mcp_runtime_active,
            timeout_sec=settings.llm.MCP_STARTUP_GREETING_TIMEOUT_SEC,
        )
    else:
        run_startup_greeting(session, mcp_runtime_active=mcp_runtime_active)
