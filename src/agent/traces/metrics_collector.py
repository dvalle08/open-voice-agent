"""Metrics collector for LiveKit agent telemetry.

Aggregates metrics from AgentSession events and publishes to data channel for
real-time monitoring. Also creates one Langfuse trace per finalized user turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from time import monotonic, time
from typing import Any, Awaitable, Callable, Optional, Sequence, Union

from livekit import rtc
from livekit.agents import metrics
from livekit.agents.telemetry import tracer  # noqa: F811 – kept at module level for monkeypatch
from opentelemetry import trace  # noqa: F401

from src.agent.traces.channel_metrics import ChannelPublisher
from src.agent.traces.turn_tracer import (
    PendingAssistantItemRecord,
    PendingUnscopedStreamRecord,
    TraceTurn,
    TurnTracer,
)
from src.core.logger import logger
from src.core.settings import settings


@dataclass
class STTMetrics:
    """Speech-to-text metrics."""

    type: str
    label: str
    request_id: str
    timestamp: float
    model_name: str
    duration: float
    audio_duration: float
    streamed: bool
    metadata: Optional[dict[str, Any]] = None


@dataclass
class LLMMetrics:
    """Language model metrics."""

    type: str
    label: str
    request_id: str
    timestamp: float
    duration: float
    ttft: float
    cancelled: bool
    completion_tokens: int
    prompt_tokens: int
    prompt_cached_tokens: int
    total_tokens: int
    tokens_per_second: float
    speech_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class TTSMetrics:
    """Text-to-speech metrics."""

    type: str
    label: str
    request_id: str
    timestamp: float
    duration: float
    ttfb: float
    audio_duration: float
    cancelled: bool
    characters_count: int
    streamed: bool
    segment_id: Optional[str] = None
    speech_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class VADMetrics:
    """Voice activity detection metrics."""

    type: str
    label: str
    timestamp: float
    idle_time: float
    inference_duration_total: float
    inference_count: int
    metadata: Optional[dict[str, Any]] = None


@dataclass
class EOUMetrics:
    """End-of-utterance metrics."""

    type: str
    timestamp: float
    end_of_utterance_delay: float
    transcription_delay: float
    on_user_turn_completed_delay: float
    speech_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class Latencies:
    """Latency breakdown for conversation turn."""

    total_latency: float
    eou_delay: float
    stt_finalization_delay: float
    llm_to_tts_handoff_latency: float
    vad_detection_delay: float
    llm_ttft: float
    tts_ttfb: float


@dataclass
class TurnMetrics:
    """Aggregated metrics for a conversation turn."""

    turn_id: str
    timestamp: float
    role: str
    transcript: str = ""
    stt: Optional[STTMetrics] = None
    eou: Optional[EOUMetrics] = None
    llm: Optional[LLMMetrics] = None
    tts: Optional[TTSMetrics] = None
    vad: Optional[VADMetrics] = None
    latencies: Optional[Latencies] = None

    def compute_latencies(
        self,
        eou_delay: float = 0.0,
        stt_finalization_delay: float = 0.0,
        observed_total_latency: Optional[float] = None,
    ) -> None:
        llm_ttft = self.llm.ttft if self.llm else 0.0
        tts_ttfb = self.tts.ttfb if self.tts else 0.0
        baseline = eou_delay + llm_ttft + tts_ttfb
        observed = observed_total_latency if observed_total_latency is not None else 0.0
        total = max(baseline, observed)
        self.latencies = Latencies(
            total_latency=total,
            eou_delay=eou_delay,
            stt_finalization_delay=stt_finalization_delay,
            llm_to_tts_handoff_latency=max(total - baseline, 0.0),
            vad_detection_delay=eou_delay,
            llm_ttft=llm_ttft,
            tts_ttfb=tts_ttfb,
        )

    def to_dict(self) -> dict:
        stt_metrics = (
            {
                **asdict(self.stt),
                "display_duration": self.stt.duration
                if self.stt.duration > 0
                else self.stt.audio_duration,
            }
            if self.stt
            else None
        )
        return {
            "type": "conversation_turn",
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "role": self.role,
            "transcript": self.transcript,
            "metrics": {
                "stt": stt_metrics,
                "eou": asdict(self.eou) if self.eou else None,
                "llm": asdict(self.llm) if self.llm else None,
                "tts": asdict(self.tts) if self.tts else None,
                "vad": asdict(self.vad) if self.vad else None,
            },
            "latencies": asdict(self.latencies) if self.latencies else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ------------------------------------------------------------------
# Consolidated speech-id keyed state
# ------------------------------------------------------------------


@dataclass
class TurnState:
    """Per-speech-id state consolidating turn metrics and timing data."""

    metrics: Optional[TurnMetrics] = None
    eou_metrics: Optional[EOUMetrics] = None
    eou_delay: float = 0.0
    stt_finalization_delay: float = 0.0
    speech_end_monotonic: Optional[float] = None
    first_audio_monotonic: Optional[float] = None
    streamed_assistant_text: str = ""
    streamed_assistant_text_observed_at: Optional[float] = None
    text_output_flush_observed: bool = False
    text_output_flush_observed_at: Optional[float] = None


@dataclass
class PendingUserUtterance:
    """Logical user utterance that may span multiple final STT chunks."""

    transcript: str
    committed: bool = False
    stt_observed: bool = False
    llm_started: bool = False
    assistant_response_started: bool = False
    assistant_response_started_at: Optional[float] = None
    llm_stalled_before_response: bool = False
    watchdog_id: Optional[str] = None
    speech_id: Optional[str] = None


@dataclass
class AssistantItemSpeechBinding:
    """Recent exact correlation between a ChatItem object and a speech id."""

    speech_id: str
    observed_at: float


@dataclass
class QueuedCollectorEvent:
    """FIFO collector event that must be processed in-order."""

    handler: Callable[..., Awaitable[Any]]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    waiter: asyncio.Future[Any] | None = None


# ------------------------------------------------------------------
# Facade
# ------------------------------------------------------------------


class MetricsCollector:
    """Collects and publishes agent metrics to LiveKit data channel."""

    UNKNOWN_SESSION_ID = "unknown-session"
    UNKNOWN_PARTICIPANT_ID = "unknown-participant"
    UNKNOWN_ROOM_ID = "unknown-room"

    def __init__(
        self,
        room: rtc.Room,
        model_name: str,
        *,
        room_name: str,
        room_id: Optional[str] = None,
        participant_id: Optional[str] = None,
        fallback_session_prefix: Optional[str] = None,
        fallback_participant_prefix: Optional[str] = None,
        langfuse_enabled: bool = False,
    ) -> None:
        self._room = room
        self._model_name = model_name
        self._room_name = room_name or self.UNKNOWN_ROOM_ID
        self._room_id = room_id or room_name or self.UNKNOWN_ROOM_ID

        fallback_session_id = _build_fallback_id(fallback_session_prefix)
        fallback_participant_id = _build_fallback_id(fallback_participant_prefix)
        session_id = fallback_session_id or self.UNKNOWN_SESSION_ID
        participant_id_resolved = (
            _normalize(participant_id)
            or fallback_participant_id
            or self.UNKNOWN_PARTICIPANT_ID
        )

        self._publisher = ChannelPublisher(room)
        self._pending_user_utterances: deque[PendingUserUtterance] = deque()
        self._pending_agent_transcripts: deque[str] = deque()
        self._unscoped_streamed_assistant_event_times: deque[float] = deque()
        self._pending_unscoped_stream_records: deque[PendingUnscopedStreamRecord] = deque()
        self._assistant_item_speech_ids: dict[int, AssistantItemSpeechBinding] = {}
        self._pending_assistant_items: dict[int, PendingAssistantItemRecord] = {}
        self._superseded_speech_ids: set[str] = set()
        self._turns: dict[str, TurnState] = {}
        self._llm_stall_tasks: dict[str, asyncio.Task[None]] = {}
        self._latest_vad_metrics: Optional[VADMetrics] = None
        self._latest_vad_metric_attributes: Optional[dict[str, Any]] = None
        self._first_final_user_turn_logged = False
        self._speech_item_callback_registered_logged = False
        self._speech_item_callback_unavailable_logged = False
        self._speech_item_callback_failed_logged = False
        self._event_queue: deque[QueuedCollectorEvent] = deque()
        self._event_worker_task: asyncio.Task[None] | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._llm_stall_timeout_sec = max(
            float(
                getattr(
                    settings.llm,
                    "TURN_LLM_STALL_TIMEOUT_SEC",
                    8.0,
                )
            ),
            0.0,
        )
        self._shutdown_drain_timeout_sec = (
            max(
                float(
                    getattr(
                        settings.langfuse,
                        "LANGFUSE_SHUTDOWN_DRAIN_TIMEOUT_MS",
                        3000.0,
                    )
                ),
                0.0,
            )
            / 1000.0
        )

        self._tracer = TurnTracer(
            publisher=self._publisher,
            room_id=self._room_id,
            session_id=session_id,
            participant_id=participant_id_resolved,
            fallback_session_id=fallback_session_id,
            fallback_participant_id=fallback_participant_id,
            langfuse_enabled=langfuse_enabled,
            pending_agent_transcripts=self._pending_agent_transcripts,
            pending_assistant_items=self._pending_assistant_items,
            unscoped_streamed_assistant_event_times=self._unscoped_streamed_assistant_event_times,
            pending_unscoped_stream_records=self._pending_unscoped_stream_records,
            superseded_speech_ids=self._superseded_speech_ids,
        )

    # Expose for tests that set collector._trace_finalize_timeout_sec directly
    @property
    def _trace_finalize_timeout_sec(self) -> float:
        return self._tracer._trace_finalize_timeout_sec

    @_trace_finalize_timeout_sec.setter
    def _trace_finalize_timeout_sec(self, value: float) -> None:
        self._tracer._trace_finalize_timeout_sec = value

    @property
    def _trace_post_tool_response_timeout_sec(self) -> float:
        return self._tracer._trace_post_tool_response_timeout_sec

    @_trace_post_tool_response_timeout_sec.setter
    def _trace_post_tool_response_timeout_sec(self, value: float) -> None:
        self._tracer._trace_post_tool_response_timeout_sec = value

    def submit_metrics_collected(self, collected_metrics: Any) -> None:
        self._submit_serialized(self._handle_metrics_collected, collected_metrics)

    def submit_user_input_transcribed(self, transcript: str, *, is_final: bool) -> None:
        self._submit_serialized(
            self._handle_user_input_transcribed,
            transcript,
            is_final=is_final,
        )

    def submit_conversation_item_added(
        self,
        *,
        role: Optional[str],
        item: Any = None,
        content: Any,
        event_created_at: Optional[float] = None,
        item_created_at: Optional[float] = None,
    ) -> None:
        self._submit_serialized(
            self._handle_conversation_item_added,
            role=role,
            item=item,
            content=content,
            event_created_at=event_created_at,
            item_created_at=item_created_at,
        )

    def submit_speech_created(self, speech_handle: Any) -> None:
        self._submit_serialized(self._handle_speech_created, speech_handle)

    def submit_function_tools_executed(
        self,
        *,
        function_calls: list[Any],
        function_call_outputs: list[Any],
        created_at: float,
    ) -> None:
        self._submit_serialized(
            self._handle_function_tools_executed,
            function_calls=function_calls,
            function_call_outputs=function_call_outputs,
            created_at=created_at,
        )

    def submit_agent_state_changed(
        self,
        *,
        old_state: str,
        new_state: str,
    ) -> None:
        self._submit_serialized(
            self._handle_agent_state_changed,
            old_state=old_state,
            new_state=new_state,
        )

    def submit_streamed_assistant_text_delta(
        self,
        speech_id: Optional[str],
        text: str,
        observed_at: float,
    ) -> None:
        self._submit_serialized(
            self._handle_streamed_assistant_text_delta,
            speech_id=speech_id,
            text=text,
            observed_at=observed_at,
        )

    def submit_streamed_assistant_text_flush(
        self,
        speech_id: Optional[str],
        observed_at: float,
    ) -> None:
        self._submit_serialized(
            self._handle_streamed_assistant_text_flush,
            speech_id=speech_id,
            observed_at=observed_at,
        )

    def submit_streamed_assistant_text_context_missing(
        self,
        observed_at: float,
    ) -> None:
        self._submit_serialized(
            self._handle_streamed_assistant_text_context_missing,
            observed_at=observed_at,
        )

    # ------------------------------------------------------------------
    # Public event handlers
    # ------------------------------------------------------------------

    async def on_session_metadata(
        self,
        *,
        session_id: Any,
        participant_id: Any,
    ) -> None:
        await self._call_serialized(
            self._handle_session_metadata,
            session_id=session_id,
            participant_id=participant_id,
        )

    async def _handle_session_metadata(
        self,
        *,
        session_id: Any,
        participant_id: Any,
    ) -> None:
        normalized_session = _normalize(session_id)
        normalized_participant = _normalize(participant_id)
        await self._tracer.on_session_metadata(
            session_id=normalized_session,
            participant_id=normalized_participant,
        )

    async def on_user_input_transcribed(
        self,
        transcript: str,
        *,
        is_final: bool,
    ) -> None:
        await self._call_serialized(
            self._handle_user_input_transcribed,
            transcript,
            is_final=is_final,
        )

    async def _handle_user_input_transcribed(
        self,
        transcript: str,
        *,
        is_final: bool,
    ) -> None:
        if not is_final:
            return
        normalized = transcript.strip()
        if not normalized:
            return
        utterance = self._current_open_user_utterance()
        if utterance is None:
            utterance = PendingUserUtterance(transcript=normalized)
            utterance.watchdog_id = self._start_llm_stall_watchdog(transcript=normalized)
            self._pending_user_utterances.append(utterance)
        else:
            if self._should_reopen_utterance_for_continuation(utterance):
                previous_speech_id = self._reopen_user_utterance_for_continuation(
                    utterance
                )
                if previous_speech_id is not None:
                    await self._tracer.recover_false_turn_before_response(
                        speech_id=previous_speech_id
                    )
            utterance.transcript = _merge_user_transcripts(
                utterance.transcript,
                normalized,
            )
            if utterance.watchdog_id is None:
                utterance.watchdog_id = self._start_llm_stall_watchdog(
                    transcript=utterance.transcript,
                )
            else:
                self._update_llm_stall_watchdog(
                    utterance.watchdog_id,
                    utterance.transcript,
                )
        if not self._first_final_user_turn_logged:
            self._first_final_user_turn_logged = True
            logger.info(
                "First finalized user transcript received: room=%s chars=%s preview=%r",
                self._room_name,
                len(utterance.transcript),
                utterance.transcript[:80],
            )
        room_id = await self._resolve_room_id()
        await self._tracer.create_turn(
            user_transcript=utterance.transcript,
            room_id=room_id,
        )

    async def on_conversation_item_added(
        self,
        *,
        role: Optional[str],
        item: Any = None,
        content: Any,
        event_created_at: Optional[float] = None,
        item_created_at: Optional[float] = None,
    ) -> None:
        await self._call_serialized(
            self._handle_conversation_item_added,
            role=role,
            item=item,
            content=content,
            event_created_at=event_created_at,
            item_created_at=item_created_at,
        )

    async def _handle_conversation_item_added(
        self,
        *,
        role: Optional[str],
        item: Any = None,
        content: Any,
        event_created_at: Optional[float] = None,
        item_created_at: Optional[float] = None,
    ) -> None:
        if role not in {"user", "assistant"}:
            return
        normalized = _extract_content_text(content).strip()
        if not normalized:
            return
        if role == "user":
            utterance = self._user_utterance_accepting_manual_update()
            if utterance is None:
                utterance = PendingUserUtterance(
                    transcript=normalized,
                    committed=True,
                )
                self._pending_user_utterances.append(utterance)
            else:
                utterance.transcript = _merge_user_transcripts(
                    utterance.transcript,
                    normalized,
                )
                utterance.committed = True
                if utterance.watchdog_id is not None:
                    self._update_llm_stall_watchdog(
                        utterance.watchdog_id,
                        utterance.transcript,
                    )
            user_event_created_at = (
                item_created_at if item_created_at is not None else event_created_at
            )
            await self._tracer.attach_user_text(
                utterance.transcript,
                event_created_at=user_event_created_at,
                speech_id=utterance.speech_id,
            )
            return
        assistant_event_created_at = (
            item_created_at if item_created_at is not None else event_created_at
        )
        self._sweep_assistant_item_caches()
        if item is not None:
            if await self._attach_assistant_item_if_correlated(
                item=item,
                assistant_text=normalized,
                event_created_at=assistant_event_created_at,
            ):
                return
            item_id = self._buffer_pending_assistant_item(
                item=item,
                assistant_text=normalized,
                event_created_at=assistant_event_created_at,
            )
            if item_id is not None:
                trace_turn = await self._tracer.try_attach_unresolved_assistant_item(
                    item_id
                )
                await self._tracer.maybe_finalize(trace_turn)
            return
        logger.debug(
            "assistant_item_fell_back_to_orphan: source=conversation_item reason=no_item_identity"
        )
        await self._on_assistant_text(
            normalized,
            event_created_at=assistant_event_created_at,
            source="conversation_item",
        )

    async def on_function_tools_executed(
        self,
        *,
        function_calls: list[Any],
        function_call_outputs: list[Any],
        created_at: float,
    ) -> None:
        await self._call_serialized(
            self._handle_function_tools_executed,
            function_calls=function_calls,
            function_call_outputs=function_call_outputs,
            created_at=created_at,
        )

    async def _handle_function_tools_executed(
        self,
        *,
        function_calls: list[Any],
        function_call_outputs: list[Any],
        created_at: float,
    ) -> None:
        trace_turn = await self._tracer.attach_function_tools_executed(
            function_calls=function_calls,
            function_call_outputs=function_call_outputs,
            created_at=created_at,
        )
        await self._publish_partial_turn_pipeline_summary(trace_turn)
        await self._tracer.maybe_finalize(trace_turn)

    async def on_tool_step_started(self) -> bool:
        return await self._call_serialized(self._handle_tool_step_started)

    async def _handle_tool_step_started(self) -> bool:
        trace_turn, should_announce = await self._tracer.attach_tool_step_started()
        await self._publish_partial_turn_pipeline_summary(trace_turn)
        await self._tracer.maybe_finalize(trace_turn)
        return should_announce

    async def on_speech_created(self, speech_handle: Any) -> None:
        await self._call_serialized(self._handle_speech_created, speech_handle)

    async def _handle_speech_created(self, speech_handle: Any) -> None:
        speech_id = _normalize(getattr(speech_handle, "id", None))
        if self._is_superseded_speech_id(speech_id):
            logger.debug(
                "Ignoring speech_created for superseded speech_id=%s",
                speech_id,
            )
            return
        if speech_id:
            self._sweep_assistant_item_caches()

        on_item_added = self._register_speech_item_added_callback(
            speech_handle=speech_handle,
            speech_id=speech_id,
        )

        for chat_item in getattr(speech_handle, "chat_items", []):
            await self._register_assistant_item_for_speech(
                item=chat_item,
                speech_id=speech_id,
            )

        assistant_text, assistant_created_at = _extract_latest_assistant_chat_item(
            getattr(speech_handle, "chat_items", [])
        )
        if assistant_text:
            await self._on_assistant_text(
                assistant_text,
                event_created_at=assistant_created_at,
                speech_id=speech_id,
                source="speech_created",
            )

        add_done_callback = getattr(speech_handle, "add_done_callback", None)
        if not callable(add_done_callback):
            return

        def _on_done(handle: Any) -> None:
            remove_item_added_callback = getattr(
                handle,
                "_remove_item_added_callback",
                None,
            )
            if callable(remove_item_added_callback) and on_item_added is not None:
                with contextlib.suppress(Exception):
                    remove_item_added_callback(on_item_added)
            done_speech_id = _normalize(getattr(handle, "id", None))
            chat_items = list(getattr(handle, "chat_items", []))
            self._submit_serialized_callback(
                self._handle_speech_done,
                done_speech_id,
                chat_items,
            )

        try:
            add_done_callback(_on_done)
        except Exception:
            return

    async def _handle_speech_item_added(
        self,
        speech_id: Optional[str],
        item: Any,
    ) -> None:
        if self._is_superseded_speech_id(speech_id):
            logger.debug(
                "Ignoring speech_item_added for superseded speech_id=%s",
                speech_id,
            )
            return
        await self._register_assistant_item_for_speech(item=item, speech_id=speech_id)

        assistant_text, event_created_at = _extract_assistant_chat_item(item)
        if not assistant_text:
            return
        await self._on_assistant_text(
            assistant_text,
            event_created_at=event_created_at,
            speech_id=speech_id,
            source="speech_item_added",
        )

    async def _handle_speech_done(
        self,
        speech_id: Optional[str],
        chat_items: list[Any],
    ) -> None:
        if self._is_superseded_speech_id(speech_id):
            logger.debug(
                "Ignoring speech_done for superseded speech_id=%s",
                speech_id,
            )
            return
        assistant_text, event_created_at = _extract_latest_assistant_chat_item(
            chat_items
        )
        trace_turn = await self._tracer.mark_speech_done(
            speech_id=speech_id,
            observed_at=event_created_at,
        )
        for chat_item in chat_items:
            await self._register_assistant_item_for_speech(
                item=chat_item,
                speech_id=speech_id,
            )
        if assistant_text:
            await self._on_assistant_text(
                assistant_text,
                event_created_at=event_created_at,
                speech_id=speech_id,
                source="speech_done",
            )
            return
        await self._tracer.maybe_finalize(trace_turn)

    async def on_agent_state_changed(
        self,
        *,
        old_state: str,
        new_state: str,
    ) -> None:
        await self._call_serialized(
            self._handle_agent_state_changed,
            old_state=old_state,
            new_state=new_state,
        )

    async def _handle_agent_state_changed(
        self,
        *,
        old_state: str,
        new_state: str,
    ) -> None:
        if new_state != "speaking":
            return
        logger.debug(
            "Ignoring agent_state_changed for tracing boundary until exact TTS arrives: old_state=%s new_state=%s",
            old_state,
            new_state,
        )

    async def _handle_streamed_assistant_text_delta(
        self,
        *,
        speech_id: Optional[str],
        text: str,
        observed_at: float,
    ) -> None:
        normalized_speech_id = _normalize(speech_id)
        if not text:
            return
        if normalized_speech_id and self._is_superseded_speech_id(normalized_speech_id):
            logger.debug(
                "Ignoring streamed assistant text delta for superseded speech_id=%s",
                normalized_speech_id,
            )
            return
        if not normalized_speech_id:
            self._buffer_unscoped_streamed_assistant_text_delta(
                text=text,
                observed_at=observed_at,
            )
            trace_turn = await self._tracer.try_attach_unscoped_streamed_assistant_text()
            await self._tracer.maybe_finalize(trace_turn)
            return
        state = self._get_or_create_state(normalized_speech_id)
        state.streamed_assistant_text = _append_assistant_text_delta(
            state.streamed_assistant_text,
            text,
        )
        state.streamed_assistant_text_observed_at = observed_at
        if not state.streamed_assistant_text.strip():
            return
        trace_turn = await self._tracer.attach_assistant_text(
            state.streamed_assistant_text,
            event_created_at=observed_at,
            speech_id=normalized_speech_id,
            source="text_output_stream",
        )
        await self._tracer.maybe_finalize(trace_turn)

    async def _handle_streamed_assistant_text_flush(
        self,
        *,
        speech_id: Optional[str],
        observed_at: float,
    ) -> None:
        normalized_speech_id = _normalize(speech_id)
        if normalized_speech_id and self._is_superseded_speech_id(normalized_speech_id):
            logger.debug(
                "Ignoring streamed assistant text flush for superseded speech_id=%s",
                normalized_speech_id,
            )
            return
        if not normalized_speech_id:
            self._mark_unscoped_streamed_assistant_text_flush(observed_at=observed_at)
            trace_turn = await self._tracer.try_attach_unscoped_streamed_assistant_text()
            await self._tracer.maybe_finalize(trace_turn)
            return
        state = self._get_or_create_state(normalized_speech_id)
        state.text_output_flush_observed = True
        state.text_output_flush_observed_at = observed_at
        trace_turn = await self._tracer.mark_streamed_assistant_text_flushed(
            speech_id=normalized_speech_id,
            observed_at=observed_at,
        )
        if state.streamed_assistant_text.strip():
            trace_turn = await self._tracer.attach_assistant_text(
                state.streamed_assistant_text,
                event_created_at=observed_at,
                speech_id=normalized_speech_id,
                source="text_output_stream",
            )
        await self._tracer.maybe_finalize(trace_turn)

    async def _handle_streamed_assistant_text_context_missing(
        self,
        *,
        observed_at: float,
    ) -> None:
        self._unscoped_streamed_assistant_event_times.append(observed_at)
        cutoff = observed_at - self._tracer._trace_legacy_finalize_timeout_sec
        while (
            self._unscoped_streamed_assistant_event_times
            and self._unscoped_streamed_assistant_event_times[0] < cutoff
        ):
            self._unscoped_streamed_assistant_event_times.popleft()
        self._sweep_unscoped_stream_records(min_observed_at=cutoff)

    async def on_metrics_collected(
        self,
        collected_metrics: Union[
            metrics.STTMetrics,
            metrics.LLMMetrics,
            metrics.TTSMetrics,
            metrics.EOUMetrics,
            metrics.VADMetrics,
        ],
    ) -> None:
        await self._call_serialized(
            self._handle_metrics_collected,
            collected_metrics,
        )

    async def _handle_metrics_collected(
        self,
        collected_metrics: Union[
            metrics.STTMetrics,
            metrics.LLMMetrics,
            metrics.TTSMetrics,
            metrics.EOUMetrics,
            metrics.VADMetrics,
        ],
    ) -> None:
        speech_id = None
        turn_metrics = None
        trace_turn: Optional[TraceTurn] = None

        if isinstance(collected_metrics, metrics.STTMetrics):
            speech_id = collected_metrics.request_id
            if self._is_superseded_speech_id(speech_id):
                logger.debug(
                    "Ignoring STT metrics for superseded speech_id=%s",
                    speech_id,
                )
                return
            turn_metrics = self._get_or_create_turn(speech_id, role="user")
            utterance = self._next_user_utterance_for_stt()
            if utterance is not None:
                turn_metrics.transcript = utterance.transcript
                utterance.stt_observed = True
            turn_metrics.stt = STTMetrics(
                type=collected_metrics.type,
                label=collected_metrics.label,
                request_id=collected_metrics.request_id,
                timestamp=collected_metrics.timestamp,
                model_name=self._model_name,
                duration=collected_metrics.duration,
                audio_duration=collected_metrics.audio_duration,
                streamed=collected_metrics.streamed,
                metadata=_metric_metadata_to_dict(collected_metrics.metadata),
            )
            await self._publish_live_update(speech_id=speech_id, stage="stt", turn_metrics=turn_metrics)
            logger.debug("STT metrics collected: request_id=%s, duration=%.3fs", speech_id, collected_metrics.duration)
            trace_turn = await self._tracer.attach_stt(
                transcript=turn_metrics.transcript,
                duration=collected_metrics.duration,
                fallback_duration=collected_metrics.audio_duration,
                metric_attributes=_stt_metric_attributes(collected_metrics),
            )

        elif isinstance(collected_metrics, metrics.LLMMetrics):
            speech_id = collected_metrics.speech_id or collected_metrics.request_id
            if self._is_superseded_speech_id(speech_id):
                logger.debug(
                    "Ignoring LLM metrics for superseded speech_id=%s",
                    speech_id,
                )
                return
            linked_utterance = self._mark_llm_stage_reached(speech_id)
            turn_metrics = self._get_or_create_turn(speech_id, role="agent")
            if linked_utterance is not None:
                turn_metrics.transcript = linked_utterance.transcript
            turn_metrics.llm = LLMMetrics(
                type=collected_metrics.type,
                label=collected_metrics.label,
                request_id=collected_metrics.request_id,
                timestamp=collected_metrics.timestamp,
                duration=collected_metrics.duration,
                ttft=collected_metrics.ttft,
                cancelled=collected_metrics.cancelled,
                completion_tokens=collected_metrics.completion_tokens,
                prompt_tokens=collected_metrics.prompt_tokens,
                prompt_cached_tokens=collected_metrics.prompt_cached_tokens,
                total_tokens=collected_metrics.total_tokens,
                tokens_per_second=collected_metrics.tokens_per_second,
                speech_id=collected_metrics.speech_id,
                metadata=_metric_metadata_to_dict(collected_metrics.metadata),
            )
            await self._publish_live_update(speech_id=speech_id, stage="llm", turn_metrics=turn_metrics)
            logger.debug("LLM metrics collected: speech_id=%s, ttft=%.3fs", speech_id, collected_metrics.ttft)
            trace_turn = await self._tracer.attach_llm(
                duration=collected_metrics.duration,
                ttft=collected_metrics.ttft,
                speech_id=speech_id,
                metric_attributes=_llm_metric_attributes(collected_metrics),
            )

        elif isinstance(collected_metrics, metrics.TTSMetrics):
            speech_id = collected_metrics.speech_id or collected_metrics.request_id
            if self._is_superseded_speech_id(speech_id):
                logger.debug(
                    "Ignoring TTS metrics for superseded speech_id=%s",
                    speech_id,
                )
                return
            state = self._get_or_create_state(speech_id)
            if state.first_audio_monotonic is None:
                state.first_audio_monotonic = monotonic()
            self._mark_assistant_response_started(
                speech_id,
                observed_at=collected_metrics.timestamp,
            )
            turn_metrics = self._get_or_create_turn(speech_id, role="agent")
            tts_metric_metadata = _metric_metadata_to_dict(collected_metrics.metadata)
            turn_metrics.tts = TTSMetrics(
                type=collected_metrics.type,
                label=collected_metrics.label,
                request_id=collected_metrics.request_id,
                timestamp=collected_metrics.timestamp,
                duration=collected_metrics.duration,
                ttfb=collected_metrics.ttfb,
                audio_duration=collected_metrics.audio_duration,
                cancelled=collected_metrics.cancelled,
                characters_count=collected_metrics.characters_count,
                streamed=collected_metrics.streamed,
                segment_id=collected_metrics.segment_id,
                speech_id=collected_metrics.speech_id,
                metadata=tts_metric_metadata,
            )
            await self._publish_live_update(speech_id=speech_id, stage="tts", turn_metrics=turn_metrics)
            logger.debug("TTS metrics collected: speech_id=%s, ttfb=%.3fs", speech_id, collected_metrics.ttfb)
            trace_turn = await self._tracer.attach_tts(
                duration=collected_metrics.duration,
                fallback_duration=collected_metrics.audio_duration,
                ttfb=collected_metrics.ttfb,
                speech_id=speech_id,
                observed_total_latency=self._observed_total_latency(speech_id),
                metric_attributes=_tts_metric_attributes(collected_metrics),
            )
            await self._tracer.mark_first_audio_started(
                speech_id=speech_id,
                started_at=collected_metrics.timestamp,
            )
            metric_assistant_text = _assistant_text_from_metadata(tts_metric_metadata)
            if metric_assistant_text:
                await self._on_assistant_text(
                    metric_assistant_text,
                    event_created_at=collected_metrics.timestamp,
                    speech_id=speech_id,
                    source="tts_metrics",
                )

        elif isinstance(collected_metrics, metrics.EOUMetrics):
            speech_id = collected_metrics.speech_id
            if speech_id:
                if self._is_superseded_speech_id(speech_id):
                    logger.debug(
                        "Ignoring EOU metrics for superseded speech_id=%s",
                        speech_id,
                    )
                    return
                linked_utterance = self._mark_oldest_open_user_utterance_committed(
                    speech_id
                )
                state = self._get_or_create_state(speech_id)
                if state.speech_end_monotonic is None:
                    state.speech_end_monotonic = monotonic()
                state.eou_delay = collected_metrics.end_of_utterance_delay
                state.stt_finalization_delay = collected_metrics.transcription_delay
                state.eou_metrics = EOUMetrics(
                    type=collected_metrics.type,
                    timestamp=collected_metrics.timestamp,
                    end_of_utterance_delay=collected_metrics.end_of_utterance_delay,
                    transcription_delay=collected_metrics.transcription_delay,
                    on_user_turn_completed_delay=collected_metrics.on_user_turn_completed_delay,
                    speech_id=collected_metrics.speech_id,
                    metadata=_metric_metadata_to_dict(collected_metrics.metadata),
                )
                turn_metrics = state.metrics
                if turn_metrics and linked_utterance is not None:
                    turn_metrics.transcript = linked_utterance.transcript
                if turn_metrics:
                    turn_metrics.eou = state.eou_metrics
                    if self._latest_vad_metrics and turn_metrics.vad is None:
                        turn_metrics.vad = self._latest_vad_metrics
                await self._publish_live_update(
                    speech_id=speech_id,
                    stage="eou",
                    turn_metrics=turn_metrics,
                )
                logger.debug("EOU metrics collected: speech_id=%s, delay=%.3fs", speech_id, collected_metrics.end_of_utterance_delay)
                trace_turn = await self._tracer.attach_eou(
                    duration=collected_metrics.end_of_utterance_delay,
                    transcription_delay=collected_metrics.transcription_delay,
                    on_user_turn_completed_delay=collected_metrics.on_user_turn_completed_delay,
                    metric_attributes=_eou_metric_attributes(collected_metrics),
                    vad_metric_attributes=self._latest_vad_metric_attributes,
                )

        elif isinstance(collected_metrics, metrics.VADMetrics):
            speech_id = getattr(collected_metrics, "speech_id", None)
            self._latest_vad_metrics = VADMetrics(
                type=collected_metrics.type,
                label=collected_metrics.label,
                timestamp=collected_metrics.timestamp,
                idle_time=collected_metrics.idle_time,
                inference_duration_total=collected_metrics.inference_duration_total,
                inference_count=collected_metrics.inference_count,
                metadata=_metric_metadata_to_dict(collected_metrics.metadata),
            )
            self._latest_vad_metric_attributes = _vad_metric_attributes(collected_metrics)
            if speech_id:
                state = self._turns.get(speech_id)
                turn_metrics = state.metrics if state else None
            if speech_id and turn_metrics:
                turn_metrics.vad = self._latest_vad_metrics
            await self._publisher.publish_live_update(
                speech_id=speech_id,
                stage="vad",
                role=turn_metrics.role if turn_metrics else None,
                turn_metrics=turn_metrics,
                vad_metrics=self._latest_vad_metrics,
                diagnostic=not bool(speech_id and turn_metrics),
                eou_delay=self._turns[speech_id].eou_delay if speech_id and speech_id in self._turns else 0.0,
                stt_finalization_delay=self._turns[speech_id].stt_finalization_delay if speech_id and speech_id in self._turns else 0.0,
                observed_total_latency=self._observed_total_latency(speech_id) if speech_id else None,
            )
            logger.debug("VAD metrics collected: speech_id=%s, idle_time=%.3fs", speech_id or "n/a", collected_metrics.idle_time)

        if speech_id and turn_metrics:
            await self._maybe_publish_turn(speech_id, turn_metrics)
        if _trace_turn_has_tool_activity(trace_turn) and isinstance(
            collected_metrics, (metrics.LLMMetrics, metrics.TTSMetrics)
        ):
            await self._publish_partial_turn_pipeline_summary(trace_turn)
        await self._tracer.maybe_finalize(trace_turn)

    async def wait_for_pending_trace_tasks(self) -> None:
        await self._wait_for_event_queue_idle()
        await self._tracer.wait_for_pending_tasks()

    async def drain_pending_traces(self) -> None:
        if self._shutdown_drain_timeout_sec <= 0.0:
            return
        await asyncio.wait_for(
            self._drain_pending_traces_once(),
            timeout=self._shutdown_drain_timeout_sec,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _drain_pending_traces_once(self) -> None:
        await self._wait_for_event_queue_idle()
        await self._tracer.drain_pending_turns()
        await self._tracer.wait_for_pending_tasks()

    async def _call_serialized(
        self,
        handler: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[Any] = loop.create_future()
        self._enqueue_serialized(handler, args=args, kwargs=kwargs, waiter=waiter)
        return await waiter

    def _submit_serialized(
        self,
        handler: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._enqueue_serialized(handler, args=args, kwargs=kwargs, waiter=None)

    def _enqueue_serialized(
        self,
        handler: Callable[..., Awaitable[Any]],
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        waiter: asyncio.Future[Any] | None,
    ) -> None:
        loop = asyncio.get_running_loop()
        if self._event_loop is None:
            self._event_loop = loop
        elif self._event_loop is not loop:
            raise RuntimeError("MetricsCollector cannot be shared across event loops")

        self._event_queue.append(
            QueuedCollectorEvent(
                handler=handler,
                args=args,
                kwargs=kwargs,
                waiter=waiter,
            )
        )
        if self._event_worker_task is None:
            self._event_worker_task = loop.create_task(self._run_event_worker())

    async def _run_event_worker(self) -> None:
        while True:
            if not self._event_queue:
                self._event_worker_task = None
                return

            event = self._event_queue.popleft()
            try:
                result = await event.handler(*event.args, **event.kwargs)
            except Exception as exc:
                if event.waiter is not None and not event.waiter.done():
                    event.waiter.set_exception(exc)
                else:
                    logger.exception(
                        "Metrics collector event processing failed: handler=%s",
                        getattr(event.handler, "__name__", repr(event.handler)),
                    )
            else:
                if event.waiter is not None and not event.waiter.done():
                    event.waiter.set_result(result)

    async def _wait_for_event_queue_idle(self) -> None:
        while self._event_worker_task is not None:
            task = self._event_worker_task
            await asyncio.gather(task, return_exceptions=True)
            if self._event_worker_task is task:
                break

    def _get_or_create_state(self, speech_id: str) -> TurnState:
        if speech_id not in self._turns:
            self._turns[speech_id] = TurnState()
        return self._turns[speech_id]

    def _get_or_create_turn(self, speech_id: str, role: str) -> TurnMetrics:
        state = self._get_or_create_state(speech_id)
        if state.metrics is None:
            state.metrics = TurnMetrics(
                turn_id=str(uuid.uuid4()),
                timestamp=time(),
                role=role,
            )
        return state.metrics

    async def _on_assistant_text(
        self,
        assistant_text: str,
        *,
        event_created_at: Optional[float] = None,
        speech_id: Optional[str] = None,
        source: str = "unknown",
    ) -> None:
        normalized = assistant_text.strip()
        if not normalized:
            return
        if speech_id is not None and self._is_superseded_speech_id(speech_id):
            logger.debug(
                "Ignoring assistant text for superseded speech_id=%s source=%s",
                speech_id,
                source,
            )
            return
        trace_turn = await self._tracer.attach_assistant_text(
            normalized,
            event_created_at=event_created_at,
            speech_id=speech_id,
            source=source,
        )
        await self._tracer.maybe_finalize(trace_turn)

    def _register_speech_item_added_callback(
        self,
        *,
        speech_handle: Any,
        speech_id: Optional[str],
    ) -> Callable[[Any], None] | None:
        add_item_added_callback = getattr(
            speech_handle,
            "_add_item_added_callback",
            None,
        )
        if not callable(add_item_added_callback):
            if not self._speech_item_callback_unavailable_logged:
                self._speech_item_callback_unavailable_logged = True
                logger.warning(
                    "SpeechHandle item-added callback unavailable; Langfuse tracing will rely on fallback sources"
                )
            return None

        def _on_item_added(item: Any) -> None:
            try:
                self._submit_serialized_callback(
                    self._handle_speech_item_added,
                    speech_id,
                    item,
                )
            except Exception:
                return

        try:
            add_item_added_callback(_on_item_added)
        except Exception as exc:
            if not self._speech_item_callback_failed_logged:
                self._speech_item_callback_failed_logged = True
                logger.warning(
                    "Failed to register SpeechHandle item-added callback; Langfuse tracing will rely on fallback sources: %s",
                    exc,
                )
            return None

        if not self._speech_item_callback_registered_logged:
            self._speech_item_callback_registered_logged = True
            logger.debug(
                "SpeechHandle item-added callback registered for provider-agnostic assistant text capture"
            )
        return _on_item_added

    async def _attach_assistant_item_if_correlated(
        self,
        *,
        item: Any,
        assistant_text: str,
        event_created_at: Optional[float],
    ) -> bool:
        item_id = _assistant_item_identity(item)
        if item_id is None:
            return False
        binding = self._assistant_item_speech_ids.get(item_id)
        if binding is None:
            return False
        if self._is_superseded_speech_id(binding.speech_id):
            self._assistant_item_speech_ids.pop(item_id, None)
            return False
        binding.observed_at = time()
        self._pending_assistant_items.pop(item_id, None)
        logger.debug(
            "assistant_item_correlated_exact: item_id=%s speech_id=%s",
            item_id,
            binding.speech_id,
        )
        await self._on_assistant_text(
            assistant_text,
            event_created_at=event_created_at,
            speech_id=binding.speech_id,
            source="conversation_item_correlated",
        )
        return True

    def _buffer_pending_assistant_item(
        self,
        *,
        item: Any,
        assistant_text: str,
        event_created_at: Optional[float],
    ) -> Optional[int]:
        item_id = _assistant_item_identity(item)
        if item_id is None:
            logger.debug(
                "assistant_item_fell_back_to_orphan: source=conversation_item reason=item_identity_unavailable"
            )
            return None
        self._pending_assistant_items[item_id] = PendingAssistantItemRecord(
            item_id=item_id,
            text=assistant_text,
            event_created_at=_to_optional_float(event_created_at),
            source="conversation_item",
            observed_at=time(),
        )
        logger.debug(
            "assistant_item_buffered_pending_speech_id: item_id=%s",
            item_id,
        )
        return item_id

    async def _register_assistant_item_for_speech(
        self,
        *,
        item: Any,
        speech_id: Optional[str],
    ) -> None:
        normalized_speech_id = _normalize_optional_str(speech_id)
        item_id = _assistant_item_identity(item)
        if normalized_speech_id is None or item_id is None:
            return
        if self._is_superseded_speech_id(normalized_speech_id):
            return
        self._assistant_item_speech_ids[item_id] = AssistantItemSpeechBinding(
            speech_id=normalized_speech_id,
            observed_at=time(),
        )
        pending_event = self._pending_assistant_items.pop(item_id, None)
        if pending_event is None:
            return
        logger.debug(
            "assistant_item_correlated_exact: item_id=%s speech_id=%s",
            item_id,
            normalized_speech_id,
        )
        assistant_text, event_created_at = _extract_assistant_chat_item(item)
        if (
            assistant_text
            and assistant_text.strip() == pending_event.text.strip()
        ):
            return
        await self._on_assistant_text(
            pending_event.text,
            event_created_at=pending_event.event_created_at,
            speech_id=normalized_speech_id,
            source="conversation_item_correlated",
        )

    def _sweep_assistant_item_caches(self) -> None:
        now = time()
        retention_sec = max(
            self._trace_finalize_timeout_sec,
            self._trace_post_tool_response_timeout_sec,
            30.0,
        )
        min_observed_at = now - retention_sec
        stale_item_ids = [
            item_id
            for item_id, binding in self._assistant_item_speech_ids.items()
            if binding.observed_at < min_observed_at
        ]
        for item_id in stale_item_ids:
            self._assistant_item_speech_ids.pop(item_id, None)

        stale_pending_ids = [
            item_id
            for item_id, event in self._pending_assistant_items.items()
            if event.observed_at < min_observed_at
        ]
        for item_id in stale_pending_ids:
            self._pending_assistant_items.pop(item_id, None)

    def _buffer_unscoped_streamed_assistant_text_delta(
        self,
        *,
        text: str,
        observed_at: float,
    ) -> None:
        if not text:
            return
        self._sweep_unscoped_stream_records(
            min_observed_at=observed_at - self._tracer._trace_legacy_finalize_timeout_sec
        )
        if (
            self._pending_unscoped_stream_records
            and not self._pending_unscoped_stream_records[-1].flush_observed
        ):
            record = self._pending_unscoped_stream_records[-1]
            record.text = _append_assistant_text_delta(record.text, text)
            record.last_delta_at = observed_at
            record.observed_at = observed_at
            return

        self._pending_unscoped_stream_records.append(
            PendingUnscopedStreamRecord(
                text=text,
                observed_at=observed_at,
                last_delta_at=observed_at,
            )
        )

    def _mark_unscoped_streamed_assistant_text_flush(
        self,
        *,
        observed_at: float,
    ) -> None:
        self._sweep_unscoped_stream_records(
            min_observed_at=observed_at - self._tracer._trace_legacy_finalize_timeout_sec
        )
        if not self._pending_unscoped_stream_records:
            return
        record = self._pending_unscoped_stream_records[-1]
        if record.flush_observed:
            return
        record.flush_observed = True
        record.flush_observed_at = observed_at
        record.observed_at = observed_at
        if record.last_delta_at is None:
            record.last_delta_at = observed_at

    def _sweep_unscoped_stream_records(
        self,
        *,
        min_observed_at: float,
    ) -> None:
        while (
            self._pending_unscoped_stream_records
            and self._pending_unscoped_stream_records[0].observed_at < min_observed_at
        ):
            self._pending_unscoped_stream_records.popleft()

    def _submit_serialized_callback(
        self,
        handler: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        loop = self._event_loop
        if loop is None or loop.is_closed():
            return

        def _enqueue() -> None:
            self._enqueue_serialized(handler, args=args, kwargs=kwargs, waiter=None)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            loop.call_soon_threadsafe(_enqueue)
            return

        if running_loop is loop:
            _enqueue()
            return

        loop.call_soon_threadsafe(_enqueue)

    async def _publish_live_update(
        self,
        *,
        speech_id: Optional[str],
        stage: str,
        turn_metrics: Optional[TurnMetrics],
    ) -> None:
        state = self._turns.get(speech_id or "") if speech_id else None
        await self._publisher.publish_live_update(
            speech_id=speech_id,
            stage=stage,
            role=turn_metrics.role if turn_metrics else None,
            turn_metrics=turn_metrics,
            eou_delay=state.eou_delay if state else 0.0,
            stt_finalization_delay=state.stt_finalization_delay if state else 0.0,
            observed_total_latency=self._observed_total_latency(speech_id or ""),
        )

    async def _publish_partial_turn_pipeline_summary(
        self,
        trace_turn: Optional[TraceTurn],
    ) -> None:
        payload = await self._tracer.build_pipeline_summary_payload(
            trace_turn,
            partial=True,
        )
        if payload is None:
            return
        await self._publisher.publish_turn_pipeline_summary(payload)

    async def _maybe_publish_turn(
        self, speech_id: str, turn_metrics: TurnMetrics
    ) -> None:
        is_complete = False
        if turn_metrics.role == "user" and turn_metrics.stt:
            is_complete = True
        elif turn_metrics.role == "agent" and turn_metrics.llm and turn_metrics.tts:
            is_complete = True

        if not is_complete:
            return

        state = self._turns.get(speech_id)
        eou_delay = state.eou_delay if state else 0.0
        stt_finalization_delay = state.stt_finalization_delay if state else 0.0
        observed = self._observed_total_latency(speech_id)
        turn_metrics.compute_latencies(
            eou_delay,
            stt_finalization_delay=stt_finalization_delay,
            observed_total_latency=observed,
        )
        await self._publisher.publish_conversation_turn(turn_metrics)
        self._turns.pop(speech_id, None)

    def _observed_total_latency(self, speech_id: str) -> Optional[float]:
        state = self._turns.get(speech_id)
        if not state:
            return None
        start = state.speech_end_monotonic
        end = state.first_audio_monotonic
        if start is None or end is None:
            return None
        if end <= start:
            return None
        return end - start

    def _is_superseded_speech_id(self, speech_id: Optional[str]) -> bool:
        normalized = _normalize(speech_id)
        if not normalized:
            return False
        return normalized in self._superseded_speech_ids

    def _quarantine_superseded_speech_id(self, speech_id: Optional[str]) -> None:
        normalized = _normalize(speech_id)
        if not normalized:
            return
        if normalized in self._superseded_speech_ids:
            return
        self._superseded_speech_ids.add(normalized)
        self._turns.pop(normalized, None)
        stale_item_ids = [
            item_id
            for item_id, binding in self._assistant_item_speech_ids.items()
            if binding.speech_id == normalized
        ]
        for item_id in stale_item_ids:
            self._assistant_item_speech_ids.pop(item_id, None)
        self._tracer.quarantine_superseded_speech_id(normalized)

    def _should_reopen_utterance_for_continuation(
        self,
        utterance: PendingUserUtterance,
    ) -> bool:
        return bool(
            utterance.speech_id is not None
            and not utterance.assistant_response_started
            and (utterance.llm_started or utterance.llm_stalled_before_response)
        )

    def _cancel_llm_stall_watchdog(self, watchdog_id: Optional[str]) -> None:
        if watchdog_id is None:
            return
        task = self._llm_stall_tasks.pop(watchdog_id, None)
        if task:
            task.cancel()

    def _reopen_user_utterance_for_continuation(
        self,
        utterance: PendingUserUtterance,
    ) -> Optional[str]:
        previous_speech_id = utterance.speech_id
        if previous_speech_id is not None:
            self._quarantine_superseded_speech_id(previous_speech_id)
        self._cancel_llm_stall_watchdog(utterance.watchdog_id)
        utterance.watchdog_id = None
        utterance.committed = False
        utterance.stt_observed = False
        utterance.llm_started = False
        utterance.llm_stalled_before_response = False
        utterance.speech_id = None
        utterance.assistant_response_started = False
        utterance.assistant_response_started_at = None
        logger.debug(
            "user_utterance_reopened_before_response_started: previous_speech_id=%s transcript_preview=%r",
            previous_speech_id,
            utterance.transcript[:80],
        )
        return previous_speech_id

    def _mark_assistant_response_started(
        self,
        speech_id: Optional[str],
        *,
        observed_at: Optional[float],
    ) -> Optional[PendingUserUtterance]:
        normalized_speech_id = _normalize(speech_id)
        if not normalized_speech_id or self._is_superseded_speech_id(normalized_speech_id):
            return None
        utterance = self._find_user_utterance_by_speech_id(
            normalized_speech_id,
            include_llm_started=True,
        )
        if utterance is None:
            return None
        utterance.llm_started = True
        utterance.assistant_response_started = True
        utterance.assistant_response_started_at = _to_optional_float(observed_at)
        self._cancel_llm_stall_watchdog(utterance.watchdog_id)
        utterance.watchdog_id = None
        self._prune_resolved_user_utterances()
        return utterance

    def _current_open_user_utterance(self) -> Optional[PendingUserUtterance]:
        utterance = self._latest_user_utterance()
        if utterance is None or utterance.assistant_response_started:
            return None
        if utterance.committed and utterance.speech_id is None:
            return None
        return utterance

    def _user_utterance_accepting_manual_update(self) -> Optional[PendingUserUtterance]:
        utterance = self._latest_user_utterance()
        if utterance is None or utterance.assistant_response_started:
            return None
        if utterance.committed and utterance.speech_id is None:
            return None
        return utterance

    def _latest_user_utterance(self) -> Optional[PendingUserUtterance]:
        if not self._pending_user_utterances:
            return None
        return self._pending_user_utterances[-1]

    def _next_user_utterance_for_stt(self) -> Optional[PendingUserUtterance]:
        for utterance in self._pending_user_utterances:
            if utterance.stt_observed:
                continue
            return utterance
        return None

    def _find_user_utterance_by_speech_id(
        self,
        speech_id: str,
        *,
        include_llm_started: bool = False,
        include_response_started: bool = True,
    ) -> Optional[PendingUserUtterance]:
        for utterance in reversed(self._pending_user_utterances):
            if utterance.speech_id != speech_id:
                continue
            if utterance.assistant_response_started and not include_response_started:
                continue
            if utterance.llm_started and not include_llm_started:
                continue
            return utterance
        return None

    def _mark_oldest_open_user_utterance_committed(
        self,
        speech_id: str,
    ) -> Optional[PendingUserUtterance]:
        linked = self._find_user_utterance_by_speech_id(
            speech_id,
            include_llm_started=True,
        )
        if linked is not None:
            linked.committed = True
            self._prune_resolved_user_utterances()
            return linked

        for utterance in self._pending_user_utterances:
            if utterance.assistant_response_started:
                continue
            if utterance.speech_id is not None and utterance.speech_id != speech_id:
                self._quarantine_superseded_speech_id(utterance.speech_id)
            utterance.committed = True
            utterance.speech_id = speech_id
            self._prune_resolved_user_utterances()
            return utterance
        return None

    def _prune_resolved_user_utterances(self) -> None:
        while self._pending_user_utterances:
            utterance = self._pending_user_utterances[0]
            if (
                not utterance.committed
                or not utterance.llm_started
                or not utterance.assistant_response_started
            ):
                break
            self._pending_user_utterances.popleft()

    def _start_llm_stall_watchdog(self, *, transcript: str) -> str | None:
        if self._llm_stall_timeout_sec <= 0:
            return None

        watchdog_id = str(uuid.uuid4())
        task = asyncio.create_task(
            self._warn_if_turn_stalled_before_llm(
                watchdog_id=watchdog_id,
                transcript=transcript,
            ),
            name=f"llm-stall-watchdog-{watchdog_id}",
        )
        self._llm_stall_tasks[watchdog_id] = task

        def _on_done(_: asyncio.Task[None]) -> None:
            self._llm_stall_tasks.pop(watchdog_id, None)

        task.add_done_callback(_on_done)
        return watchdog_id

    def _update_llm_stall_watchdog(self, watchdog_id: str, transcript: str) -> None:
        utterance = self._find_user_utterance_by_watchdog(watchdog_id)
        if utterance is None or utterance.assistant_response_started:
            return
        utterance.transcript = transcript

    def _mark_llm_stage_reached(
        self,
        speech_id: Optional[str],
    ) -> Optional[PendingUserUtterance]:
        normalized_speech_id = _normalize(speech_id)
        utterance: Optional[PendingUserUtterance] = None
        if normalized_speech_id is not None:
            if self._is_superseded_speech_id(normalized_speech_id):
                return None
            utterance = self._find_user_utterance_by_speech_id(
                normalized_speech_id,
                include_llm_started=True,
            )

        if utterance is None:
            for candidate in self._pending_user_utterances:
                if candidate.assistant_response_started:
                    continue
                if candidate.llm_started and candidate.speech_id is not None:
                    continue
                utterance = candidate
                break

        if utterance is None:
            return None

        if normalized_speech_id is not None and utterance.speech_id is None:
            utterance.speech_id = normalized_speech_id
        utterance.llm_started = True
        watchdog_id = utterance.watchdog_id
        utterance.watchdog_id = None
        if watchdog_id is not None:
            self._cancel_llm_stall_watchdog(watchdog_id)
        return utterance

    def _find_user_utterance_by_watchdog(
        self,
        watchdog_id: str,
    ) -> Optional[PendingUserUtterance]:
        for utterance in self._pending_user_utterances:
            if utterance.watchdog_id == watchdog_id:
                return utterance
        return None

    async def _warn_if_turn_stalled_before_llm(
        self,
        *,
        watchdog_id: str,
        transcript: str,
    ) -> None:
        try:
            await asyncio.sleep(self._llm_stall_timeout_sec)
        except asyncio.CancelledError:
            return

        utterance = self._find_user_utterance_by_watchdog(watchdog_id)
        if utterance is None or utterance.llm_started:
            return

        utterance.watchdog_id = None
        utterance.llm_stalled_before_response = True
        preview = utterance.transcript[:80] if utterance.transcript else transcript[:80]
        logger.warning(
            "Turn stalled before LLM stage: timeout=%.2fs room=%s transcript_chars=%s transcript_preview=%r",
            self._llm_stall_timeout_sec,
            self._room_name,
            len(utterance.transcript or transcript),
            preview,
        )

    async def _resolve_room_id(self) -> str:
        if self._room_id and self._room_id != self._room_name:
            return self._room_id
        try:
            sid = await self._room.sid
            normalized = _normalize(sid)
            if normalized:
                self._room_id = normalized
        except Exception:
            pass
        return self._room_id or self._room_name or self.UNKNOWN_ROOM_ID


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _normalize(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _build_fallback_id(prefix: Optional[str]) -> Optional[str]:
    normalized = _normalize(prefix)
    if not normalized:
        return None
    return f"{normalized}_{uuid.uuid4()}"


def _append_if_new(queue: deque[str], value: str) -> None:
    if queue and queue[-1] == value:
        return
    queue.append(value)


def _append_assistant_text_delta(existing: str, incoming: str) -> str:
    if not incoming:
        return existing
    if not existing:
        return incoming
    return f"{existing}{incoming}"


def _merge_user_transcripts(existing: str, incoming: str) -> str:
    left = existing.strip()
    right = incoming.strip()
    if not left:
        return right
    if not right:
        return left
    if left == right:
        return left
    if left.casefold() in right.casefold():
        return right
    if right.casefold() in left.casefold():
        return left
    if right.startswith(left):
        return right
    if left.startswith(right):
        return left

    left_words = left.split()
    right_words = right.split()
    max_overlap = min(len(left_words), len(right_words))
    for overlap in range(max_overlap, 0, -1):
        left_suffix = [word.casefold() for word in left_words[-overlap:]]
        right_prefix = [word.casefold() for word in right_words[:overlap]]
        if left_suffix == right_prefix:
            merged_words = [*left_words, *right_words[overlap:]]
            return " ".join(merged_words).strip()
    return f"{left} {right}".strip()


def _trace_turn_has_tool_activity(trace_turn: Optional[TraceTurn]) -> bool:
    if trace_turn is None:
        return False
    return bool(
        trace_turn.tool_step_announced
        or trace_turn.tool_phase_open
        or trace_turn.last_tool_event_order is not None
        or trace_turn.tool_executions
    )


def _extract_content_text(content: Any) -> str:
    """Extract plain text from a chat content payload.

    Handles LiveKit's known content structures: str, list[str],
    list[{text: str}], and objects with a .text attribute or method.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (bytes, bytearray)):
        return content.decode("utf-8", errors="ignore")
    if isinstance(content, dict):
        for key in ("text", "content", "parts", "message", "output", "value", "delta"):
            if key in content:
                return _extract_content_text(content[key])
        return ""
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts = [_extract_content_text(item) for item in content]
        return " ".join(p.strip() for p in parts if p.strip())

    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if callable(text_attr):
        try:
            result = text_attr()
            if isinstance(result, str):
                return result
        except Exception:
            pass

    for attr_name in ("content", "parts", "message", "output", "delta"):
        nested = getattr(content, attr_name, None)
        if nested is not None:
            result = _extract_content_text(nested)
            if result:
                return result
    return ""


def _extract_latest_assistant_chat_item(chat_items: Any) -> tuple[str, Optional[float]]:
    """Extract latest assistant text and created_at from speech handle chat items."""
    if isinstance(chat_items, (str, bytes, bytearray)) or not isinstance(
        chat_items, Sequence
    ):
        return "", None
    latest_text = ""
    latest_created_at: Optional[float] = None
    for item in chat_items:
        role = getattr(item, "role", None)
        if isinstance(role, str) and role not in {"assistant"}:
            continue
        content = getattr(item, "content", None)
        text = _extract_content_text(content)
        normalized = text.strip()
        if not normalized:
            continue
        latest_text = normalized
        latest_created_at = _to_optional_float(getattr(item, "created_at", None))
    return latest_text, latest_created_at


def _extract_assistant_chat_item(item: Any) -> tuple[str, Optional[float]]:
    role = getattr(item, "role", None)
    if isinstance(role, str) and role != "assistant":
        return "", None
    normalized = _extract_content_text(getattr(item, "content", None)).strip()
    if not normalized:
        return "", None
    return normalized, _to_optional_float(getattr(item, "created_at", None))


def _assistant_item_identity(item: Any) -> Optional[int]:
    if item is None:
        return None
    return id(item)


def _normalize_optional_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _to_optional_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _metric_metadata_to_dict(metadata: Any) -> Optional[dict[str, Any]]:
    if metadata is None:
        return None
    if hasattr(metadata, "model_dump"):
        dumped = metadata.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
        return {"value": dumped}
    if isinstance(metadata, dict):
        return metadata
    return {"value": str(metadata)}


def _assistant_text_from_metadata(metadata: Optional[dict[str, Any]]) -> str:
    if not metadata:
        return ""
    for key in ("assistant_text", "spoken_text"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _metadata_attributes(metadata: Any) -> dict[str, Any]:
    data = _metric_metadata_to_dict(metadata)
    if not data:
        return {}
    return _flatten_attributes(data, prefix="metadata")


def _flatten_attributes(
    data: dict[str, Any], *, prefix: str = ""
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if value is None:
            continue
        if isinstance(value, dict):
            flattened.update(_flatten_attributes(value, prefix=full_key))
            continue
        if isinstance(value, (list, tuple)):
            serialized = [_safe_attr_value(v) for v in value]
            flattened[full_key] = serialized
            continue
        flattened[full_key] = _safe_attr_value(value)
    return flattened


def _safe_attr_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _stt_metric_attributes(collected_metrics: metrics.STTMetrics) -> dict[str, Any]:
    attrs = {
        "type": collected_metrics.type,
        "label": collected_metrics.label,
        "request_id": collected_metrics.request_id,
        "timestamp": collected_metrics.timestamp,
        "duration": collected_metrics.duration,
        "audio_duration": collected_metrics.audio_duration,
        "streamed": collected_metrics.streamed,
    }
    attrs.update(_metadata_attributes(collected_metrics.metadata))
    return attrs


def _eou_metric_attributes(collected_metrics: metrics.EOUMetrics) -> dict[str, Any]:
    attrs = {
        "type": collected_metrics.type,
        "timestamp": collected_metrics.timestamp,
        "end_of_utterance_delay": collected_metrics.end_of_utterance_delay,
        "transcription_delay": collected_metrics.transcription_delay,
        "on_user_turn_completed_delay": collected_metrics.on_user_turn_completed_delay,
        "speech_id": collected_metrics.speech_id,
    }
    attrs.update(_metadata_attributes(collected_metrics.metadata))
    return attrs


def _vad_metric_attributes(collected_metrics: metrics.VADMetrics) -> dict[str, Any]:
    attrs = {
        "type": collected_metrics.type,
        "label": collected_metrics.label,
        "timestamp": collected_metrics.timestamp,
        "idle_time": collected_metrics.idle_time,
        "inference_duration_total": collected_metrics.inference_duration_total,
        "inference_count": collected_metrics.inference_count,
    }
    attrs.update(_metadata_attributes(collected_metrics.metadata))
    return attrs


def _llm_metric_attributes(collected_metrics: metrics.LLMMetrics) -> dict[str, Any]:
    attrs = {
        "type": collected_metrics.type,
        "label": collected_metrics.label,
        "request_id": collected_metrics.request_id,
        "timestamp": collected_metrics.timestamp,
        "duration": collected_metrics.duration,
        "ttft": collected_metrics.ttft,
        "cancelled": collected_metrics.cancelled,
        "completion_tokens": collected_metrics.completion_tokens,
        "prompt_tokens": collected_metrics.prompt_tokens,
        "prompt_cached_tokens": collected_metrics.prompt_cached_tokens,
        "total_tokens": collected_metrics.total_tokens,
        "tokens_per_second": collected_metrics.tokens_per_second,
        "speech_id": collected_metrics.speech_id,
    }
    attrs.update(_metadata_attributes(collected_metrics.metadata))
    return attrs


def _tts_metric_attributes(collected_metrics: metrics.TTSMetrics) -> dict[str, Any]:
    attrs = {
        "type": collected_metrics.type,
        "label": collected_metrics.label,
        "request_id": collected_metrics.request_id,
        "timestamp": collected_metrics.timestamp,
        "ttfb": collected_metrics.ttfb,
        "duration": collected_metrics.duration,
        "audio_duration": collected_metrics.audio_duration,
        "cancelled": collected_metrics.cancelled,
        "characters_count": collected_metrics.characters_count,
        "streamed": collected_metrics.streamed,
        "segment_id": collected_metrics.segment_id,
        "speech_id": collected_metrics.speech_id,
    }
    attrs.update(_metadata_attributes(collected_metrics.metadata))
    return attrs
