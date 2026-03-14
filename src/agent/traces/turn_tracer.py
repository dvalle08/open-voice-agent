"""Langfuse OTel turn tracer for LiveKit agent telemetry.

Creates one OpenTelemetry trace per finalized user turn and publishes
a ``trace_update`` message to the data channel when emission completes.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from time import time, time_ns
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

from opentelemetry import trace

from src.core.logger import logger
from src.core.settings import settings

if TYPE_CHECKING:
    from src.agent.traces.channel_metrics import ChannelPublisher


@dataclass
class TraceTurn:
    """Langfuse turn trace payload."""

    turn_id: str
    session_id: str
    room_id: str
    participant_id: str
    user_transcript: str
    prompt_text: str
    created_at: float = field(default_factory=time)
    user_turn_committed: bool = False
    user_turn_committed_at: Optional[float] = None
    user_transcript_updated_at: Optional[float] = None
    response_text: str = ""
    assistant_text: str = ""
    assistant_text_missing: bool = False
    stt_status: str = "missing"
    vad_duration_ms: Optional[float] = None
    stt_duration_ms: Optional[float] = None
    stt_finalization_ms: Optional[float] = None
    stt_total_latency_ms: Optional[float] = None
    eou_on_user_turn_completed_ms: Optional[float] = None
    speech_id: Optional[str] = None
    llm_duration_ms: Optional[float] = None
    llm_ttft_ms: Optional[float] = None
    llm_total_latency_ms: Optional[float] = None
    tts_duration_ms: Optional[float] = None
    tts_ttfb_ms: Optional[float] = None
    tts_total_latency_ms: Optional[float] = None
    conversational_latency_ms: Optional[float] = None
    perceived_latency_first_audio_ms: Optional[float] = None
    perceived_latency_second_audio_ms: Optional[float] = None
    llm_to_tts_handoff_ms: Optional[float] = None
    stt_attributes: dict[str, Any] = field(default_factory=dict)
    eou_attributes: dict[str, Any] = field(default_factory=dict)
    vad_attributes: dict[str, Any] = field(default_factory=dict)
    llm_attributes: dict[str, Any] = field(default_factory=dict)
    tts_attributes: dict[str, Any] = field(default_factory=dict)
    llm_calls: list["LLMCallTrace"] = field(default_factory=list)
    tts_calls: list["TTSCallTrace"] = field(default_factory=list)
    tool_executions: list["ToolExecutionTrace"] = field(default_factory=list)
    timeline_events: list["TimelineEvent"] = field(default_factory=list)
    tool_step_announced: bool = False
    tool_phase_open: bool = False
    last_tool_event_at: Optional[float] = None
    last_tool_event_order: Optional[int] = None
    last_tool_completed_at: Optional[float] = None
    assistant_text_updated_at: Optional[float] = None
    assistant_text_updated_order: Optional[int] = None
    tts_updated_at: Optional[float] = None
    tts_updated_order: Optional[int] = None
    event_counter: int = 0
    tool_post_response_missing: bool = False
    assistant_audio_started: bool = False
    assistant_audio_started_at: Optional[float] = None
    interrupted: bool = False
    interrupted_reason: Optional[str] = None
    finalization_reason: Optional[str] = None
    assistant_text_source: Optional[str] = None
    emit_ready_at: Optional[float] = None
    orphan_assistant_cutoff_at: Optional[float] = None
    coalesced_turn_ids: list[str] = field(default_factory=list)
    coalesced_user_transcripts: list[str] = field(default_factory=list)
    coalesced_fragment_count: int = 0
    trace_id: Optional[str] = None


@dataclass
class ToolCallTrace:
    """Per-tool call trace payload."""

    name: str
    call_id: str
    arguments: str
    output: str
    is_error: bool
    created_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[float] = None


@dataclass
class ToolExecutionTrace:
    """Tool execution batch trace payload."""

    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    observed_order: int = 0
    observed_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class LLMCallTrace:
    """Per-LLM-call trace payload."""

    duration_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    speech_id: Optional[str] = None
    observed_order: int = 0


@dataclass
class TTSCallTrace:
    """Per-TTS-call trace payload."""

    duration_ms: Optional[float] = None
    ttfb_ms: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    speech_id: Optional[str] = None
    assistant_text: str = ""
    observed_order: int = 0
    first_audio_at: Optional[float] = None


@dataclass
class AssistantTextRecord:
    """Buffered assistant text that has not been correlated safely yet."""

    text: str
    event_created_at: Optional[float] = None
    source: Optional[str] = None


@dataclass
class TimelineEvent:
    """Ordered event for building trace phases."""

    kind: Literal["llm", "tts", "tool_execution"]
    ref_index: int
    order: int


@dataclass
class ResponsePhaseBlock:
    """Grouped response phase block."""

    index: int
    llm_calls: list[LLMCallTrace] = field(default_factory=list)
    tts_calls: list[TTSCallTrace] = field(default_factory=list)


@dataclass
class ToolExecutionBlock:
    """Grouped tool execution block."""

    index: int
    execution: ToolExecutionTrace


_DEFAULT_TRACE_FINALIZE_TIMEOUT_MS = 8000.0
_DEFAULT_ASSISTANT_TEXT_GRACE_TIMEOUT_MS = 500.0
_DEFAULT_POST_TOOL_RESPONSE_TIMEOUT_MS = 30000.0
_DEFAULT_MAX_PENDING_TRACE_TASKS = 200
_DEFAULT_TRACE_FLUSH_TIMEOUT_SEC = 1.0
_DEFAULT_CONTINUATION_COALESCE_WINDOW_MS = 1500.0
_TOOL_ERROR_FALLBACK_TEXT = "I couldn't complete that tool request. Please rephrase the query."


class TurnTracer:
    """Creates one Langfuse trace per finalized user turn."""

    UNKNOWN_SESSION_ID = "unknown-session"
    UNKNOWN_PARTICIPANT_ID = "unknown-participant"
    UNKNOWN_ROOM_ID = "unknown-room"

    def __init__(
        self,
        *,
        publisher: ChannelPublisher,
        room_id: str,
        session_id: str,
        participant_id: str,
        fallback_session_id: Optional[str],
        fallback_participant_id: Optional[str],
        langfuse_enabled: bool,
        pending_agent_transcripts: deque[str],
    ) -> None:
        self._publisher = publisher
        self._room_id = room_id
        self._session_id = session_id
        self._participant_id = participant_id
        self._fallback_session_id = fallback_session_id
        self._fallback_participant_id = fallback_participant_id
        self._langfuse_enabled = langfuse_enabled
        self._pending_agent_transcripts = pending_agent_transcripts

        self._pending_trace_turns: deque[TraceTurn] = deque()
        self._pending_agent_transcripts_by_speech_id: dict[
            str, deque[AssistantTextRecord]
        ] = {}
        self._orphan_assistant_text_records: deque[AssistantTextRecord] = deque()
        self._trace_lock = asyncio.Lock()
        self._trace_emit_tasks: set[asyncio.Task[None]] = set()
        self._trace_finalize_tasks: dict[str, asyncio.Task[None]] = {}
        self._trace_finalize_task_versions: dict[str, int] = {}

        assistant_text_grace_timeout_ms = float(
            getattr(
                settings.langfuse,
                "LANGFUSE_ASSISTANT_TEXT_GRACE_TIMEOUT_MS",
                _DEFAULT_ASSISTANT_TEXT_GRACE_TIMEOUT_MS,
            )
        )
        self._trace_finalize_timeout_sec = (
            max(
                assistant_text_grace_timeout_ms,
                0.0,
            )
            / 1000.0
        )
        self._trace_legacy_finalize_timeout_sec = (
            max(
                getattr(
                    settings.langfuse,
                    "LANGFUSE_TRACE_FINALIZE_TIMEOUT_MS",
                    _DEFAULT_TRACE_FINALIZE_TIMEOUT_MS,
                ),
                0.0,
            )
            / 1000.0
        )
        self._trace_post_tool_response_timeout_sec = (
            max(
                getattr(
                    settings.langfuse,
                    "LANGFUSE_POST_TOOL_RESPONSE_TIMEOUT_MS",
                    _DEFAULT_POST_TOOL_RESPONSE_TIMEOUT_MS,
                ),
                0.0,
            )
            / 1000.0
        )
        self._trace_max_pending_tasks = max(
            int(
                getattr(
                    settings.langfuse,
                    "LANGFUSE_MAX_PENDING_TRACE_TASKS",
                    _DEFAULT_MAX_PENDING_TRACE_TASKS,
                )
            ),
            1,
        )
        self._trace_flush_timeout_sec = (
            max(
                getattr(
                    settings.langfuse,
                    "LANGFUSE_TRACE_FLUSH_TIMEOUT_MS",
                    _DEFAULT_TRACE_FLUSH_TIMEOUT_SEC * 1000.0,
                ),
                0.0,
            )
            / 1000.0
        )
        self._continuation_coalesce_window_sec = (
            max(
                getattr(
                    settings.langfuse,
                    "LANGFUSE_CONTINUATION_COALESCE_WINDOW_MS",
                    _DEFAULT_CONTINUATION_COALESCE_WINDOW_MS,
                ),
                0.0,
            )
            / 1000.0
        )

    # ------------------------------------------------------------------
    # Session context
    # ------------------------------------------------------------------

    async def on_session_metadata(
        self,
        *,
        session_id: Optional[str],
        participant_id: Optional[str],
    ) -> None:
        async with self._trace_lock:
            previous_session_id = self._session_id
            previous_participant_id = self._participant_id
            if session_id:
                self._session_id = session_id
            if participant_id:
                self._participant_id = participant_id

            if (
                self._session_id != previous_session_id
                or self._participant_id != previous_participant_id
            ):
                logger.debug(
                    "Session context updated: session_id=%s participant_id=%s",
                    self._session_id,
                    self._participant_id,
                )

            for turn in self._pending_trace_turns:
                if (
                    turn.session_id
                    in {self.UNKNOWN_SESSION_ID, self._fallback_session_id}
                    and self._session_id
                    not in {self.UNKNOWN_SESSION_ID, self._fallback_session_id}
                ):
                    turn.session_id = self._session_id
                if (
                    turn.participant_id
                    in {self.UNKNOWN_PARTICIPANT_ID, self._fallback_participant_id}
                    and self._participant_id
                    not in {
                        self.UNKNOWN_PARTICIPANT_ID,
                        self._fallback_participant_id,
                    }
                ):
                    turn.participant_id = self._participant_id

    # ------------------------------------------------------------------
    # Turn creation
    # ------------------------------------------------------------------

    async def create_turn(self, *, user_transcript: str, room_id: str) -> None:
        completed_turns: list[TraceTurn] = []
        timeout_schedules: list[tuple[str, float]] = []
        async with self._trace_lock:
            normalized = user_transcript.strip()
            if not normalized:
                return

            (
                completed_turns,
                timeout_schedules,
            ) = self._finalize_interrupted_turns_before_new_user_turn_locked()

            current_turn = self._latest_turn_where(
                self._turn_accepting_additional_user_input
            )
            if current_turn is not None:
                self._update_user_turn_text(current_turn, normalized)
            else:
                new_turn = TraceTurn(
                    turn_id=str(uuid.uuid4()),
                    session_id=self._session_id,
                    room_id=room_id,
                    participant_id=self._participant_id,
                    user_transcript=normalized,
                    prompt_text=normalized,
                )
                new_turn.user_transcript_updated_at = new_turn.created_at

                coalesced_turn = self._coalesced_turn_candidate()
                if coalesced_turn is not None:
                    self._absorb_coalesced_turn_metadata(new_turn, coalesced_turn)
                    self._pending_trace_turns.remove(coalesced_turn)
                    self._cancel_finalize_timeout(coalesced_turn.turn_id)

                self._pending_trace_turns.append(new_turn)

        for completed_turn in completed_turns:
            self._schedule_trace_emit(completed_turn)
        for turn_id, timeout_sec in timeout_schedules:
            self._schedule_finalize_timeout(turn_id, timeout_sec)

    async def attach_user_text(
        self,
        user_transcript: str,
        *,
        event_created_at: Optional[float] = None,
        speech_id: Optional[str] = None,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            normalized_speech_id = _normalize_optional_str(speech_id)
            turn: Optional[TraceTurn] = None
            if normalized_speech_id:
                turn = self._latest_turn_where(
                    lambda c: c.speech_id == normalized_speech_id
                )
                if turn is None:
                    turn = self._latest_turn_where(
                        lambda c: c.speech_id is None and not c.llm_calls
                    )
                    if turn is not None:
                        turn.speech_id = normalized_speech_id
                if turn is not None:
                    self._absorb_pending_pre_llm_turns(turn)

            if turn is None:
                turn = self._latest_turn_where(
                    self._turn_accepting_additional_user_input
                )
            if turn is None:
                turn = self._latest_turn_where(lambda c: not c.assistant_text.strip())
            if turn is None:
                turn = self._latest_turn_where(lambda _: True)
            if turn is None:
                return None

            normalized = user_transcript.strip()
            if not normalized:
                return turn

            self._update_user_turn_text(
                turn,
                normalized,
                event_created_at=event_created_at,
            )
            self._maybe_mark_emit_ready(turn)
            turn.user_turn_committed = True
            turn.user_turn_committed_at = _resolved_event_timestamp(
                _to_optional_float(event_created_at)
            )
            return turn

    # ------------------------------------------------------------------
    # Stage attachment
    # ------------------------------------------------------------------

    async def attach_stt(
        self,
        *,
        transcript: str,
        duration: float,
        fallback_duration: float,
        metric_attributes: Optional[dict[str, Any]] = None,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_where(
                lambda c: c.stt_status != "measured"
            )
            if not turn:
                return None
            if transcript:
                turn.user_transcript = transcript.strip()
            turn.prompt_text = turn.user_transcript
            measured_ms = _duration_to_ms_or_none(duration, fallback_duration)
            if measured_ms is None:
                turn.stt_duration_ms = None
                if turn.stt_total_latency_ms is None:
                    turn.stt_status = "missing"
            else:
                turn.stt_duration_ms = measured_ms
                turn.stt_status = "measured"
            turn.stt_attributes = _sanitize_component_attributes(metric_attributes)
            _recompute_perceived_first_audio_latency(turn)
            return turn

    async def attach_eou(
        self,
        *,
        duration: float,
        transcription_delay: float,
        on_user_turn_completed_delay: float = 0.0,
        metric_attributes: Optional[dict[str, Any]] = None,
        vad_metric_attributes: Optional[dict[str, Any]] = None,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_where(
                lambda c: c.vad_duration_ms is None
            )
            if not turn:
                return None
            eou_delay_ms = _duration_to_ms(duration, 0.0)
            turn.vad_duration_ms = eou_delay_ms
            turn.stt_finalization_ms = _duration_to_ms(transcription_delay, 0.0)
            turn.stt_total_latency_ms = eou_delay_ms + (turn.stt_finalization_ms or 0.0)
            turn.eou_on_user_turn_completed_ms = _duration_to_ms(
                on_user_turn_completed_delay, 0.0
            )
            if turn.stt_total_latency_ms > 0:
                turn.stt_status = "measured"
                if turn.stt_duration_ms is None:
                    turn.stt_duration_ms = turn.stt_total_latency_ms
            turn.eou_attributes = _sanitize_component_attributes(metric_attributes)
            turn.user_turn_committed = True
            turn.user_turn_committed_at = _resolved_event_timestamp(
                _to_optional_float(turn.eou_attributes.get("timestamp"))
            )
            metric_speech_id = _normalize_optional_str(
                turn.eou_attributes.get("speech_id")
            )
            if metric_speech_id and turn.speech_id is None:
                turn.speech_id = metric_speech_id
            turn.vad_attributes = _sanitize_component_attributes(vad_metric_attributes)
            _recompute_perceived_first_audio_latency(turn)
            return turn

    async def attach_llm(
        self,
        *,
        duration: float,
        ttft: float,
        speech_id: Optional[str] = None,
        metric_attributes: Optional[dict[str, Any]] = None,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            normalized_speech_id = _normalize_optional_str(speech_id)
            turn = self._select_turn_for_llm(normalized_speech_id)
            if not turn:
                return None
            turn.prompt_text = turn.user_transcript
            if normalized_speech_id and turn.speech_id is None:
                turn.speech_id = normalized_speech_id
            if normalized_speech_id:
                self._apply_buffered_assistant_text_for_speech_id(turn)

            llm_attrs = _sanitize_component_attributes(metric_attributes)
            order = self._next_event_order(turn)
            llm_call = LLMCallTrace(
                duration_ms=_duration_to_ms(duration, 0.0),
                ttft_ms=_duration_to_ms(ttft, 0.0),
                attributes=llm_attrs,
                speech_id=normalized_speech_id,
                observed_order=order,
            )
            turn.llm_calls.append(llm_call)
            turn.timeline_events.append(
                TimelineEvent(
                    kind="llm",
                    ref_index=len(turn.llm_calls) - 1,
                    order=order,
                )
            )

            turn.llm_attributes = llm_attrs
            if turn.llm_duration_ms is None:
                turn.llm_duration_ms = llm_call.duration_ms
            if turn.llm_ttft_ms is None:
                turn.llm_ttft_ms = llm_call.ttft_ms
            turn.llm_total_latency_ms = _sum_llm_duration_ms(turn.llm_calls)
            self._maybe_mark_emit_ready(turn)
            _recompute_perceived_first_audio_latency(turn)
            return turn

    async def attach_tts(
        self,
        *,
        duration: float,
        fallback_duration: float,
        ttfb: float,
        speech_id: Optional[str] = None,
        observed_total_latency: Optional[float],
        metric_attributes: Optional[dict[str, Any]] = None,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            normalized_speech_id = _normalize_optional_str(speech_id)
            turn = self._select_turn_for_tts(normalized_speech_id)
            if not turn:
                return None

            if normalized_speech_id and turn.speech_id is None:
                turn.speech_id = normalized_speech_id
            if normalized_speech_id:
                self._apply_buffered_assistant_text_for_speech_id(turn)

            tts_attrs = _sanitize_component_attributes(metric_attributes)
            tts_metric_assistant_text = _assistant_text_from_component_attributes(tts_attrs)
            if tts_metric_assistant_text and (
                not turn.assistant_text.strip()
                or turn.assistant_text_source == "tts_metrics"
            ):
                self._apply_assistant_text_to_turn(
                    turn,
                    tts_metric_assistant_text,
                    event_created_at=_to_optional_float(tts_attrs.get("timestamp")),
                    source="tts_metrics",
                )
            order = self._next_event_order(turn)
            tts_call = TTSCallTrace(
                duration_ms=_duration_to_ms(duration, fallback_duration),
                ttfb_ms=_duration_to_ms(ttfb, 0.0),
                attributes=tts_attrs,
                speech_id=normalized_speech_id,
                assistant_text=turn.assistant_text or turn.response_text,
                observed_order=order,
                first_audio_at=_estimate_tts_first_audio_at(
                    metric_attributes=tts_attrs,
                    duration_sec=duration if duration > 0 else fallback_duration,
                    ttfb_sec=max(ttfb, 0.0),
                ),
            )
            turn.tts_calls.append(tts_call)
            turn.timeline_events.append(
                TimelineEvent(
                    kind="tts",
                    ref_index=len(turn.tts_calls) - 1,
                    order=order,
                )
            )

            turn.tts_attributes = tts_attrs
            if turn.tts_duration_ms is None:
                turn.tts_duration_ms = tts_call.duration_ms
            if turn.tts_ttfb_ms is None:
                turn.tts_ttfb_ms = tts_call.ttfb_ms
            turn.tts_total_latency_ms = _sum_tts_duration_ms(turn.tts_calls)
            tts_event_created_at = _to_optional_float(tts_attrs.get("timestamp"))
            turn.tts_updated_at = _resolved_event_timestamp(tts_event_created_at)
            turn.tts_updated_order = order

            self._maybe_mark_emit_ready(turn)
            _recompute_perceived_first_audio_latency(turn)
            if observed_total_latency is not None and len(turn.tts_calls) == 1:
                observed_ms = observed_total_latency * 1000.0
                baseline_ms = turn.perceived_latency_first_audio_ms
                turn.perceived_latency_first_audio_ms = max(
                    observed_ms,
                    baseline_ms if baseline_ms is not None else 0.0,
                )
                turn.conversational_latency_ms = turn.perceived_latency_first_audio_ms
            turn.llm_to_tts_handoff_ms = _compute_llm_to_tts_handoff_ms(
                total_latency_ms=turn.perceived_latency_first_audio_ms,
                vad_duration_ms=turn.vad_duration_ms,
                llm_ttft_ms=turn.llm_ttft_ms,
                tts_ttfb_ms=turn.tts_ttfb_ms,
            )
            self._maybe_update_perceived_second_audio_latency(turn, tts_call)
            self._maybe_close_tool_phase(turn)
            return turn

    async def attach_assistant_text(
        self,
        assistant_text: str,
        *,
        event_created_at: Optional[float] = None,
        speech_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            normalized_text = assistant_text.strip()
            if not normalized_text:
                return None

            normalized_speech_id = _normalize_optional_str(speech_id)
            resolved_event_created_at = _to_optional_float(event_created_at)

            if normalized_speech_id:
                turn = self._resolve_turn_for_exact_speech_id(normalized_speech_id)
                if turn is None:
                    self._buffer_assistant_text(
                        normalized_text,
                        event_created_at=resolved_event_created_at,
                        speech_id=normalized_speech_id,
                        source=source,
                    )
                    return None
                if (
                    source == "tts_metrics"
                    and turn.assistant_text.strip()
                    and turn.assistant_text_source not in {None, "tts_metrics", "unavailable"}
                ):
                    return turn
                if (
                    source == "speech_done"
                    and turn.assistant_text.strip()
                    and turn.assistant_text_source == "speech_item_added"
                ):
                    return turn
                self._apply_assistant_text_to_turn(
                    turn,
                    normalized_text,
                    event_created_at=resolved_event_created_at,
                    source=source,
                )
                return turn

            turn = self._select_turn_for_orphan_assistant_text(
                event_created_at=resolved_event_created_at
            )
            if turn is None:
                self._buffer_assistant_text(
                    normalized_text,
                    event_created_at=resolved_event_created_at,
                    source=source,
                )
                return None

            if (
                source == "tts_metrics"
                and turn.assistant_text.strip()
                and turn.assistant_text_source not in {None, "tts_metrics", "unavailable"}
            ):
                return turn
            self._apply_assistant_text_to_turn(
                turn,
                normalized_text,
                event_created_at=resolved_event_created_at,
                source=source,
            )
            return turn

    async def attach_tool_step_started(self) -> tuple[Optional[TraceTurn], bool]:
        async with self._trace_lock:
            turn = self._latest_turn_where(lambda c: bool(c.llm_calls))
            if not turn:
                turn = self._latest_turn_where(lambda _: True)
            if not turn:
                return None, False

            should_announce = not turn.tool_phase_open
            turn.tool_step_announced = True
            turn.tool_phase_open = True
            return turn, should_announce

    async def attach_function_tools_executed(
        self,
        *,
        function_calls: list[Any],
        function_call_outputs: list[Any],
        created_at: float,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._latest_turn_where(
                lambda c: bool(c.llm_calls) or c.tool_phase_open
            )
            if not turn:
                return None

            event_created_at = _to_optional_float(created_at)
            turn.tool_step_announced = True
            turn.tool_phase_open = True
            turn.tool_post_response_missing = False
            order = self._next_event_order(turn)
            turn.last_tool_event_at = _resolved_event_timestamp(event_created_at)
            turn.last_tool_event_order = order
            resolved_completed_at = turn.last_tool_event_at
            batch_calls: list[ToolCallTrace] = []
            for function_call, function_call_output in zip(
                function_calls, function_call_outputs, strict=False
            ):
                call_trace = _build_tool_call_trace(
                    function_call=function_call,
                    function_call_output=function_call_output,
                    event_created_at=event_created_at,
                )
                batch_calls.append(call_trace)
                call_completed_at = _resolved_event_timestamp(call_trace.completed_at)
                if call_completed_at > resolved_completed_at:
                    resolved_completed_at = call_completed_at

            turn.last_tool_completed_at = resolved_completed_at
            turn.tool_executions.append(
                ToolExecutionTrace(
                    tool_calls=batch_calls,
                    observed_order=order,
                    observed_at=turn.last_tool_event_at,
                    completed_at=resolved_completed_at,
                )
            )
            turn.timeline_events.append(
                TimelineEvent(
                    kind="tool_execution",
                    ref_index=len(turn.tool_executions) - 1,
                    order=order,
                )
            )

            self._maybe_close_tool_phase(turn)
            return turn

    async def mark_first_audio_started(
        self,
        *,
        speech_id: str,
        started_at: Optional[float] = None,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            normalized_speech_id = _normalize_optional_str(speech_id)
            if not normalized_speech_id:
                return None
            turn = self._resolve_turn_for_exact_speech_id(normalized_speech_id)
            if turn is None:
                return None
            turn.assistant_audio_started = True
            turn.assistant_audio_started_at = _resolved_event_timestamp(
                _to_optional_float(started_at)
            )
            return turn

    async def drain_pending_turns(self) -> None:
        completed_turns: list[TraceTurn] = []
        async with self._trace_lock:
            for turn in list(self._pending_trace_turns):
                self._apply_buffered_assistant_text_for_speech_id(turn)
                self._try_attach_latest_usable_orphan_assistant_text(turn)
                if not (turn.user_transcript and turn.llm_calls and turn.tts_calls):
                    continue
                requires_post_tool_response = self._requires_post_tool_follow_up(turn)
                missing_post_tool_assistant = bool(
                    requires_post_tool_response
                    and not self._post_tool_assistant_observed(turn)
                )
                if turn.interrupted_reason is None:
                    turn.interrupted = True
                    turn.interrupted_reason = "shutdown_drain"
                completed_turns.append(
                    self._finalize_locked(
                        turn,
                        missing_assistant_fallback=(
                            missing_post_tool_assistant or not bool(turn.assistant_text)
                        ),
                        tool_post_response_missing=requires_post_tool_response,
                        drop_assistant_text=missing_post_tool_assistant,
                    )
                )

        for completed_turn in completed_turns:
            self._schedule_trace_emit(completed_turn)

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    async def maybe_finalize(self, trace_turn: Optional[TraceTurn]) -> None:
        if not trace_turn:
            return

        completed_turn: Optional[TraceTurn] = None
        schedule_timeout_for_turn: Optional[tuple[str, float]] = None
        async with self._trace_lock:
            if trace_turn not in self._pending_trace_turns:
                return
            if not self._is_complete(trace_turn):
                if self._should_schedule_finalize_timeout(trace_turn):
                    schedule_timeout_for_turn = (
                        trace_turn.turn_id,
                        self._resolve_finalize_timeout_sec(trace_turn),
                    )
            else:
                completed_turn = self._finalize_locked(trace_turn)

        if schedule_timeout_for_turn:
            self._schedule_finalize_timeout(*schedule_timeout_for_turn)
        if completed_turn:
            self._schedule_trace_emit(completed_turn)

    async def wait_for_pending_tasks(self) -> None:
        tasks = list(self._trace_emit_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_user_turn_text(
        self,
        turn: TraceTurn,
        user_transcript: str,
        *,
        event_created_at: Optional[float] = None,
    ) -> None:
        normalized = user_transcript.strip()
        if not normalized:
            return
        merged = _merge_user_transcripts(turn.user_transcript, normalized)
        turn.user_transcript = merged
        turn.prompt_text = merged
        turn.user_transcript_updated_at = _resolved_event_timestamp(
            _to_optional_float(event_created_at)
        )

    def _coalesced_turn_candidate(self) -> Optional[TraceTurn]:
        if self._continuation_coalesce_window_sec <= 0.0:
            return None

        now = time()
        for turn in reversed(self._pending_trace_turns):
            if not self._can_coalesce_turn(turn):
                continue
            activity_at = self._turn_recent_activity_at(turn)
            if activity_at is None:
                continue
            if now - activity_at > self._continuation_coalesce_window_sec:
                return None
            return turn
        return None

    def _can_coalesce_turn(self, turn: TraceTurn) -> bool:
        if not turn.user_turn_committed:
            return False
        if not turn.user_transcript.strip():
            return False
        if turn.assistant_audio_started:
            return False
        if turn.assistant_text.strip() or turn.response_text.strip():
            return False
        if turn.tool_step_announced or turn.tool_executions or turn.last_tool_event_order is not None:
            return False
        return bool(turn.llm_calls and turn.tts_calls)

    def _turn_recent_activity_at(self, turn: TraceTurn) -> Optional[float]:
        candidates = [
            turn.assistant_text_updated_at,
            turn.tts_updated_at,
            turn.last_tool_completed_at,
            turn.last_tool_event_at,
            turn.user_turn_committed_at,
            turn.user_transcript_updated_at,
            turn.created_at,
        ]
        resolved = [candidate for candidate in candidates if candidate is not None]
        if not resolved:
            return None
        return max(resolved)

    def _absorb_coalesced_turn_metadata(
        self,
        new_turn: TraceTurn,
        absorbed_turn: TraceTurn,
    ) -> None:
        combined_input = _merge_user_transcripts(
            absorbed_turn.user_transcript,
            new_turn.user_transcript,
        )
        new_turn.user_transcript = combined_input
        new_turn.prompt_text = combined_input
        new_turn.coalesced_turn_ids = [
            *absorbed_turn.coalesced_turn_ids,
            absorbed_turn.turn_id,
        ]
        new_turn.coalesced_user_transcripts = [
            *absorbed_turn.coalesced_user_transcripts,
            absorbed_turn.user_transcript,
        ]
        new_turn.coalesced_fragment_count = (
            absorbed_turn.coalesced_fragment_count + 1
        )
        absorbed_recent_activity = self._turn_recent_activity_at(absorbed_turn)
        existing_cutoff = new_turn.orphan_assistant_cutoff_at
        if absorbed_recent_activity is not None:
            new_turn.orphan_assistant_cutoff_at = max(
                existing_cutoff or absorbed_recent_activity,
                absorbed_recent_activity,
            )

    def _next_turn_where(
        self,
        predicate: Callable[[TraceTurn], bool],
    ) -> Optional[TraceTurn]:
        for turn in self._pending_trace_turns:
            if predicate(turn):
                return turn
        return None

    def _latest_turn_where(
        self,
        predicate: Callable[[TraceTurn], bool],
    ) -> Optional[TraceTurn]:
        for turn in reversed(self._pending_trace_turns):
            if predicate(turn):
                return turn
        return None

    def _next_event_order(self, turn: TraceTurn) -> int:
        turn.event_counter += 1
        return turn.event_counter

    def _is_emit_ready(self, turn: TraceTurn) -> bool:
        return bool(turn.user_transcript and turn.llm_calls and turn.tts_calls)

    def _maybe_mark_emit_ready(self, turn: TraceTurn) -> None:
        if turn.emit_ready_at is not None:
            return
        if not self._is_emit_ready(turn):
            return
        turn.emit_ready_at = time()

    def _turn_accepting_additional_user_input(self, turn: TraceTurn) -> bool:
        if turn.llm_calls or turn.tts_calls:
            return False
        if turn.assistant_text.strip() or turn.response_text.strip():
            return False
        if turn.interrupted:
            return False
        if not turn.user_turn_committed:
            return True
        return turn.speech_id is not None

    def _absorb_pending_pre_llm_turns(self, anchor_turn: TraceTurn) -> None:
        if anchor_turn not in self._pending_trace_turns:
            return
        try:
            anchor_index = self._pending_trace_turns.index(anchor_turn)
        except ValueError:
            return

        anchor_speech_id = _normalize_optional_str(anchor_turn.speech_id)
        absorbed_turns: list[TraceTurn] = []
        for candidate in list(self._pending_trace_turns)[anchor_index + 1 :]:
            candidate_speech_id = _normalize_optional_str(candidate.speech_id)
            if candidate.llm_calls or candidate.tts_calls:
                continue
            if candidate.assistant_text.strip() or candidate.response_text.strip():
                continue
            if candidate.tool_step_announced or candidate.tool_executions:
                continue
            if candidate.last_tool_event_order is not None:
                continue
            if candidate.interrupted:
                continue
            if candidate_speech_id not in {None, anchor_speech_id}:
                continue
            if candidate.user_transcript.strip():
                self._update_user_turn_text(
                    anchor_turn,
                    candidate.user_transcript,
                    event_created_at=candidate.user_transcript_updated_at,
                )
            anchor_turn.user_turn_committed = (
                anchor_turn.user_turn_committed or candidate.user_turn_committed
            )
            if candidate.user_turn_committed_at is not None:
                anchor_turn.user_turn_committed_at = max(
                    anchor_turn.user_turn_committed_at or candidate.user_turn_committed_at,
                    candidate.user_turn_committed_at,
                )
            absorbed_turns.append(candidate)

        for absorbed_turn in absorbed_turns:
            self._pending_trace_turns.remove(absorbed_turn)
            self._cancel_finalize_timeout(absorbed_turn.turn_id)

    def _select_turn_for_llm(self, speech_id: Optional[str]) -> Optional[TraceTurn]:
        if speech_id:
            matched = self._latest_turn_where(lambda c: c.speech_id == speech_id)
            if matched:
                self._absorb_pending_pre_llm_turns(matched)
                return matched
            matched = self._latest_turn_where(
                lambda c: c.speech_id is None and not c.llm_calls
            )
            if matched is not None:
                matched.speech_id = speech_id
                self._absorb_pending_pre_llm_turns(matched)
                return matched
        matched = self._next_turn_where(lambda c: not c.llm_calls)
        if matched is not None:
            self._absorb_pending_pre_llm_turns(matched)
        return matched

    def _select_turn_for_tts(self, speech_id: Optional[str]) -> Optional[TraceTurn]:
        if speech_id:
            matched = self._latest_turn_where(lambda c: c.speech_id == speech_id)
            if matched:
                return matched
        matched_without_id = self._latest_turn_where(
            lambda c: c.speech_id is None and bool(c.llm_calls)
        )
        if matched_without_id:
            return matched_without_id
        return self._latest_turn_where(lambda c: bool(c.llm_calls))

    def _resolve_turn_for_exact_speech_id(self, speech_id: str) -> Optional[TraceTurn]:
        matched = self._latest_turn_where(lambda c: c.speech_id == speech_id)
        if matched is not None:
            return matched
        candidates = [
            turn
            for turn in self._pending_trace_turns
            if turn.speech_id is None and bool(turn.llm_calls or turn.tts_calls)
        ]
        if len(candidates) != 1:
            return None
        turn = candidates[0]
        turn.speech_id = speech_id
        return turn

    def _select_turn_for_orphan_assistant_text(
        self,
        *,
        event_created_at: Optional[float],
    ) -> Optional[TraceTurn]:
        candidates = self._assistant_text_correlation_candidates()
        if len(candidates) == 1:
            turn = candidates[0]
        else:
            emit_ready_candidates = [
                turn
                for turn in candidates
                if self._is_emit_ready(turn) and self._emit_ready_turn_is_recent(turn)
            ]
            if len(emit_ready_candidates) != 1:
                return None
            turn = emit_ready_candidates[0]
        cutoff = turn.orphan_assistant_cutoff_at
        if cutoff is not None:
            if event_created_at is None:
                return None
            if event_created_at < cutoff:
                return None
        return turn

    def _emit_ready_turn_is_recent(self, turn: TraceTurn) -> bool:
        if turn.emit_ready_at is None:
            return False
        recent_window_sec = max(self._trace_finalize_timeout_sec, 1.0)
        return (time() - turn.emit_ready_at) <= recent_window_sec

    def _apply_assistant_text_to_turn(
        self,
        turn: TraceTurn,
        assistant_text: str,
        *,
        event_created_at: Optional[float],
        source: Optional[str],
    ) -> None:
        previous_assistant_text = turn.assistant_text or turn.response_text
        order = self._next_event_order(turn)
        turn.assistant_text = assistant_text
        turn.response_text = assistant_text
        turn.assistant_text_missing = False
        turn.assistant_text_source = source or turn.assistant_text_source or "unknown"
        turn.assistant_text_updated_at = _resolved_event_timestamp(event_created_at)
        turn.assistant_text_updated_order = order
        _reconcile_assistant_text_with_tts_calls(
            turn=turn,
            assistant_text=assistant_text,
            previous_assistant_text=previous_assistant_text,
        )
        self._maybe_close_tool_phase(turn)

    def _buffer_assistant_text(
        self,
        assistant_text: str,
        *,
        event_created_at: Optional[float],
        speech_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> None:
        normalized = assistant_text.strip()
        if not normalized:
            return
        record = AssistantTextRecord(
            text=normalized,
            event_created_at=_to_optional_float(event_created_at),
            source=source,
        )
        normalized_speech_id = _normalize_optional_str(speech_id)
        if normalized_speech_id:
            queue = self._pending_agent_transcripts_by_speech_id.setdefault(
                normalized_speech_id,
                deque(),
            )
            if queue and queue[-1].text == normalized:
                return
            queue.append(record)
            return
        if self._orphan_assistant_text_records and self._orphan_assistant_text_records[-1].text == normalized:
            return
        self._orphan_assistant_text_records.append(record)

    def _apply_buffered_assistant_text_for_speech_id(self, turn: TraceTurn) -> None:
        speech_id = _normalize_optional_str(turn.speech_id)
        if not speech_id:
            return
        queue = self._pending_agent_transcripts_by_speech_id.get(speech_id)
        if not queue:
            return
        while queue:
            record = queue.popleft()
            self._apply_assistant_text_to_turn(
                turn,
                record.text,
                event_created_at=record.event_created_at,
                source=record.source or "buffered_exact",
            )
        if not queue:
            self._pending_agent_transcripts_by_speech_id.pop(speech_id, None)

    def _try_attach_latest_usable_orphan_assistant_text(
        self,
        turn: TraceTurn,
    ) -> bool:
        if not self._orphan_assistant_text_records:
            return False
        if self._select_turn_for_orphan_assistant_text(
            event_created_at=self._orphan_assistant_text_records[-1].event_created_at
        ) is not turn:
            return False
        for index in range(len(self._orphan_assistant_text_records) - 1, -1, -1):
            record = self._orphan_assistant_text_records[index]
            if self._select_turn_for_orphan_assistant_text(
                event_created_at=record.event_created_at
            ) is not turn:
                continue
            del self._orphan_assistant_text_records[index]
            self._apply_assistant_text_to_turn(
                turn,
                record.text,
                event_created_at=record.event_created_at,
                source=record.source or "orphan_buffer",
            )
            return True
        return False

    def _maybe_update_perceived_second_audio_latency(
        self,
        turn: TraceTurn,
        tts_call: TTSCallTrace,
    ) -> None:
        tool_order = turn.last_tool_event_order
        if tool_order is None:
            return
        if tts_call.observed_order <= tool_order:
            return

        observed_ms: Optional[float] = None
        if turn.last_tool_completed_at is not None and tts_call.first_audio_at is not None:
            observed_ms = max(
                (tts_call.first_audio_at - turn.last_tool_completed_at) * 1000.0,
                0.0,
            )

        fallback_ms = max(tts_call.ttfb_ms or 0.0, 0.0)
        post_tool_llm = _latest_llm_call_after_order(
            turn.llm_calls,
            min_order=tool_order,
            max_order=tts_call.observed_order,
        )
        if post_tool_llm is not None:
            fallback_ms += max(post_tool_llm.ttft_ms or 0.0, 0.0)

        if observed_ms is not None:
            turn.perceived_latency_second_audio_ms = max(observed_ms, fallback_ms)
            return
        turn.perceived_latency_second_audio_ms = fallback_ms

    def _is_complete(self, turn: TraceTurn) -> bool:
        base_complete = bool(self._is_emit_ready(turn) and turn.assistant_text)
        if not base_complete:
            return False
        if turn.tool_phase_open:
            return False
        return not self._requires_post_tool_response(turn)

    def _should_schedule_finalize_timeout(self, turn: TraceTurn) -> bool:
        return bool(
            self._is_emit_ready(turn)
            and not self._is_complete(turn)
            and not (turn.tool_phase_open and turn.last_tool_event_at is None)
            and self._resolve_finalize_timeout_sec(turn) > 0.0
        )

    def _resolve_finalize_timeout_sec(self, turn: TraceTurn) -> float:
        if self._requires_post_tool_response(turn):
            return self._trace_post_tool_response_timeout_sec
        return self._trace_finalize_timeout_sec

    def _finalize_wait_reason(self, turn: TraceTurn) -> str:
        if self._requires_post_tool_response(turn):
            return "post_tool_response"
        return "assistant_text_grace"

    def _requires_post_tool_response(self, turn: TraceTurn) -> bool:
        if not turn.tool_step_announced and turn.last_tool_event_order is None:
            return False
        return not self._post_tool_response_observed(turn)

    def _post_tool_assistant_observed(self, turn: TraceTurn) -> bool:
        if turn.last_tool_event_order is None:
            return False
        return self._observed_after_last_tool(
            turn=turn,
            event_at=turn.assistant_text_updated_at,
            event_order=turn.assistant_text_updated_order,
        )

    def _observed_after_last_tool(
        self,
        *,
        turn: TraceTurn,
        event_at: Optional[float],
        event_order: Optional[int],
    ) -> bool:
        if turn.last_tool_event_order is None:
            return False
        if turn.last_tool_event_at is not None and event_at is not None:
            if event_at > turn.last_tool_event_at:
                return True
            if event_at < turn.last_tool_event_at:
                return False
        return bool(
            event_order is not None
            and event_order > turn.last_tool_event_order
        )

    def _post_tool_response_observed(self, turn: TraceTurn) -> bool:
        if turn.last_tool_event_order is None:
            return False
        assistant_seen = self._post_tool_assistant_observed(turn)
        tts_seen = self._observed_after_last_tool(
            turn=turn,
            event_at=turn.tts_updated_at,
            event_order=turn.tts_updated_order,
        )
        return assistant_seen and tts_seen

    def _maybe_close_tool_phase(self, turn: TraceTurn) -> None:
        if not turn.tool_phase_open:
            return
        if not self._post_tool_response_observed(turn):
            return
        turn.tool_phase_open = False

    def _finalize_locked(
        self,
        turn: TraceTurn,
        *,
        missing_assistant_fallback: bool = False,
        tool_post_response_missing: bool = False,
        drop_assistant_text: bool = False,
        finalization_reason: Optional[str] = None,
    ) -> TraceTurn:
        if drop_assistant_text:
            turn.assistant_text = ""
            turn.response_text = ""

        if not turn.prompt_text:
            turn.prompt_text = turn.user_transcript
        if not turn.response_text and turn.assistant_text:
            turn.response_text = turn.assistant_text
        if not turn.assistant_text and turn.response_text:
            turn.assistant_text = turn.response_text

        if missing_assistant_fallback and not turn.assistant_text:
            fallback_text, fallback_source = self._best_available_assistant_text(
                turn,
                min_observed_order=(
                    turn.last_tool_event_order if tool_post_response_missing else None
                ),
                include_pending_agent_transcripts=not tool_post_response_missing,
            )
            if fallback_text:
                self._apply_assistant_text_to_turn(
                    turn,
                    fallback_text,
                    event_created_at=None,
                    source=fallback_source or "unknown",
                )
            else:
                tool_error_fallback = ""
                if tool_post_response_missing:
                    tool_error_fallback = _tool_error_fallback_text(turn)

                if tool_error_fallback:
                    self._apply_assistant_text_to_turn(
                        turn,
                        tool_error_fallback,
                        event_created_at=None,
                        source="tool_fallback",
                    )
                else:
                    turn.assistant_text_missing = True
                    unavailable = "[assistant text unavailable]"
                    turn.assistant_text = unavailable
                    turn.assistant_text_source = "unavailable"
                    if not turn.response_text:
                        turn.response_text = unavailable
                    logger.warning(
                        "Langfuse turn finalized without assistant text: turn_id=%s speech_id=%s reason=%s",
                        turn.turn_id,
                        turn.speech_id,
                        finalization_reason
                        or ("post_tool_timeout" if tool_post_response_missing else "assistant_text_grace_timeout"),
                    )

        turn.tool_phase_open = False
        if tool_post_response_missing:
            turn.tool_post_response_missing = True
        if finalization_reason is None:
            if turn.interrupted_reason == "shutdown_drain":
                finalization_reason = "shutdown_drain"
            elif turn.interrupted and turn.assistant_audio_started:
                finalization_reason = "interrupted_after_audio"
            elif tool_post_response_missing:
                finalization_reason = "post_tool_timeout"
            elif missing_assistant_fallback:
                finalization_reason = "assistant_text_grace_timeout"
            else:
                finalization_reason = "complete"
        turn.finalization_reason = finalization_reason

        self._pending_trace_turns.remove(turn)
        self._cancel_finalize_timeout(turn.turn_id)
        return turn

    def _best_available_assistant_text(
        self,
        turn: TraceTurn,
        *,
        min_observed_order: Optional[int] = None,
        include_pending_agent_transcripts: bool = True,
    ) -> tuple[str, Optional[str]]:
        speech_id = _normalize_optional_str(turn.speech_id)
        if turn.assistant_text.strip():
            if (
                min_observed_order is None
                or (
                    turn.assistant_text_updated_order is not None
                    and turn.assistant_text_updated_order > min_observed_order
                )
            ):
                return turn.assistant_text.strip(), turn.assistant_text_source
        if turn.response_text.strip():
            if (
                min_observed_order is None
                or (
                    turn.assistant_text_updated_order is not None
                    and turn.assistant_text_updated_order > min_observed_order
                )
            ):
                return turn.response_text.strip(), turn.assistant_text_source
        if speech_id:
            buffered_exact = self._pending_agent_transcripts_by_speech_id.get(speech_id)
            if buffered_exact:
                while buffered_exact:
                    record = buffered_exact.popleft()
                    if not record.text.strip():
                        continue
                    if not buffered_exact:
                        self._pending_agent_transcripts_by_speech_id.pop(speech_id, None)
                    return record.text.strip(), record.source or "buffered_exact"
        for tts_call in reversed(turn.tts_calls):
            if (
                min_observed_order is not None
                and tts_call.observed_order <= min_observed_order
            ):
                continue
            if tts_call.assistant_text.strip():
                return tts_call.assistant_text.strip(), "tts_metrics"
        if self._try_attach_latest_usable_orphan_assistant_text(turn):
            if turn.assistant_text.strip():
                return turn.assistant_text.strip(), turn.assistant_text_source
        if (
            include_pending_agent_transcripts
            and self._pending_agent_transcripts
            and self._can_consume_pending_agent_transcript_for_turn(turn)
        ):
            return self._pending_agent_transcripts.popleft().strip(), "pending_agent_transcript"
        return "", None

    def _assistant_text_correlation_candidates(self) -> list[TraceTurn]:
        return [
            turn
            for turn in self._pending_trace_turns
            if bool(turn.llm_calls or turn.tts_calls or turn.tool_phase_open)
            and not (turn.interrupted and turn.assistant_audio_started)
        ]

    def _can_consume_pending_agent_transcript_for_turn(self, turn: TraceTurn) -> bool:
        candidates = self._assistant_text_correlation_candidates()
        return len(candidates) == 1 and candidates[0] is turn

    def _has_active_finalize_timeout(self, turn_id: str) -> bool:
        task = self._trace_finalize_tasks.get(turn_id)
        return task is not None and not task.done()

    def _finalize_interrupted_turns_before_new_user_turn_locked(
        self,
    ) -> tuple[list[TraceTurn], list[tuple[str, float]]]:
        completed_turns: list[TraceTurn] = []
        timeout_schedules: list[tuple[str, float]] = []
        for turn in list(self._pending_trace_turns):
            if not (turn.user_transcript and turn.llm_calls and turn.tts_calls):
                continue
            if not turn.assistant_audio_started:
                continue
            turn.interrupted = True
            if not turn.interrupted_reason:
                turn.interrupted_reason = "user_barge_in_after_audio_started"
            requires_post_tool_response = self._requires_post_tool_follow_up(turn)
            missing_post_tool_assistant = bool(
                requires_post_tool_response and not self._post_tool_assistant_observed(turn)
            )
            if not turn.assistant_text.strip() or missing_post_tool_assistant:
                fallback_text, fallback_source = self._best_available_assistant_text(
                    turn,
                    min_observed_order=(
                        turn.last_tool_event_order if missing_post_tool_assistant else None
                    ),
                    include_pending_agent_transcripts=not missing_post_tool_assistant,
                )
                if fallback_text:
                    self._apply_assistant_text_to_turn(
                        turn,
                        fallback_text,
                        event_created_at=None,
                        source=fallback_source or "unknown",
                    )
                    missing_post_tool_assistant = False

            if turn.assistant_text.strip() and not missing_post_tool_assistant:
                completed_turns.append(
                    self._finalize_locked(
                        turn,
                        missing_assistant_fallback=False,
                        tool_post_response_missing=False,
                        drop_assistant_text=False,
                    )
                )
                continue

            timeout_sec = self._resolve_finalize_timeout_sec(turn)
            if timeout_sec <= 0.0:
                completed_turns.append(
                    self._finalize_locked(
                        turn,
                        missing_assistant_fallback=True,
                        tool_post_response_missing=requires_post_tool_response,
                        drop_assistant_text=missing_post_tool_assistant,
                    )
                )
                continue
            if self._has_active_finalize_timeout(turn.turn_id):
                continue
            timeout_schedules.append((turn.turn_id, timeout_sec))
        return completed_turns, timeout_schedules

    def _requires_post_tool_follow_up(self, turn: TraceTurn) -> bool:
        if turn.last_tool_event_order is None:
            return False
        return self._requires_post_tool_response(turn)

    # ------------------------------------------------------------------
    # Timeout scheduling
    # ------------------------------------------------------------------

    def _schedule_finalize_timeout(self, turn_id: str, timeout_sec: float) -> None:
        if timeout_sec <= 0.0:
            return

        version = self._trace_finalize_task_versions.get(turn_id, 0) + 1
        self._trace_finalize_task_versions[turn_id] = version

        existing_task = self._trace_finalize_tasks.get(turn_id)
        current = asyncio.current_task()
        if existing_task and not existing_task.done() and existing_task is not current:
            existing_task.cancel()

        task = asyncio.create_task(
            self._finalize_after_timeout(
                turn_id=turn_id,
                version=version,
                timeout_sec=timeout_sec,
            )
        )
        self._trace_finalize_tasks[turn_id] = task
        turn = next((t for t in self._pending_trace_turns if t.turn_id == turn_id), None)
        if turn is not None:
            logger.debug(
                "Scheduled Langfuse finalize wait: turn_id=%s speech_id=%s timeout_sec=%.3f wait_reason=%s",
                turn.turn_id,
                turn.speech_id,
                timeout_sec,
                self._finalize_wait_reason(turn),
            )
        task.add_done_callback(
            lambda _task, tid=turn_id, v=version: self._on_finalize_timeout_task_done(
                turn_id=tid,
                version=v,
            )
        )

    def _on_finalize_timeout_task_done(self, *, turn_id: str, version: int) -> None:
        if self._trace_finalize_task_versions.get(turn_id) != version:
            return
        self._trace_finalize_tasks.pop(turn_id, None)

    def _cancel_finalize_timeout(self, turn_id: str) -> None:
        self._trace_finalize_task_versions.pop(turn_id, None)
        task = self._trace_finalize_tasks.pop(turn_id, None)
        current = asyncio.current_task()
        if task and not task.done() and task is not current:
            task.cancel()

    async def _finalize_after_timeout(
        self,
        *,
        turn_id: str,
        version: int,
        timeout_sec: float,
    ) -> None:
        await asyncio.sleep(timeout_sec)

        completed_turn: Optional[TraceTurn] = None
        async with self._trace_lock:
            if self._trace_finalize_task_versions.get(turn_id) != version:
                return

            pending_turn = next(
                (t for t in self._pending_trace_turns if t.turn_id == turn_id),
                None,
            )
            if not pending_turn:
                return

            if self._is_complete(pending_turn):
                completed_turn = self._finalize_locked(pending_turn)
            elif (
                pending_turn.user_transcript
                and pending_turn.llm_calls
                and pending_turn.tts_calls
            ):
                requires_post_tool_response = self._requires_post_tool_response(
                    pending_turn
                )
                missing_post_tool_assistant = bool(
                    requires_post_tool_response
                    and not self._post_tool_assistant_observed(pending_turn)
                )
                completed_turn = self._finalize_locked(
                    pending_turn,
                    missing_assistant_fallback=(
                        missing_post_tool_assistant
                        or not bool(pending_turn.assistant_text)
                    ),
                    tool_post_response_missing=requires_post_tool_response,
                    drop_assistant_text=missing_post_tool_assistant,
                )

        if completed_turn:
            self._schedule_trace_emit(completed_turn)


    # ------------------------------------------------------------------
    # Trace emission
    # ------------------------------------------------------------------

    def _schedule_trace_emit(self, turn: TraceTurn) -> None:
        if len(self._trace_emit_tasks) >= self._trace_max_pending_tasks:
            logger.warning(
                "Dropping Langfuse trace for turn_id=%s due to pending trace backlog (%s)",
                turn.turn_id,
                len(self._trace_emit_tasks),
            )
            return
        task = asyncio.create_task(self._emit_and_publish(turn))
        self._trace_emit_tasks.add(task)
        task.add_done_callback(lambda t: self._trace_emit_tasks.discard(t))

    async def _emit_and_publish(self, turn: TraceTurn) -> None:
        await self._publisher.publish_turn_pipeline_summary(
            _build_turn_pipeline_summary(turn, partial=False)
        )
        await self._emit_turn_trace(turn)
        if turn.trace_id:
            await self._publisher.publish_trace_update(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
            )

    async def build_pipeline_summary_payload(
        self,
        trace_turn: Optional[TraceTurn],
        *,
        partial: bool,
    ) -> Optional[dict[str, Any]]:
        if trace_turn is None:
            return None

        async with self._trace_lock:
            if partial and trace_turn not in self._pending_trace_turns:
                return None
            return _build_turn_pipeline_summary(trace_turn, partial=partial)

    async def _emit_turn_trace(self, turn: TraceTurn) -> None:
        if not self._langfuse_enabled:
            return

        import src.agent.traces.metrics_collector as _mc

        _tracer = _mc.tracer

        try:
            vals = _prepare_span_values(turn)
            phase_blocks = _build_phase_blocks(turn)
            last_response_phase_index = _last_response_phase_index(phase_blocks)
            trace_output = _build_trace_output(
                turn=turn,
                phase_blocks=phase_blocks,
            )

            root_context = trace.set_span_in_context(trace.INVALID_SPAN)
            root_start_ns = time_ns()
            cursor_ns = root_start_ns

            turn_span = _tracer.start_span(
                "turn", context=root_context, start_time=root_start_ns
            )
            try:
                turn.trace_id = trace.format_trace_id(
                    turn_span.get_span_context().trace_id
                )
                _set_root_attributes(turn_span, turn, vals, trace_output)

                ctx = trace.set_span_in_context(turn_span)
                user_start_ns = cursor_ns
                user_span = _tracer.start_span(
                    "user_input", context=ctx, start_time=user_start_ns
                )
                user_cursor_ns = user_start_ns
                try:
                    _set_observation_attributes(
                        user_span,
                        input_text=turn.user_transcript,
                        output_text=turn.user_transcript,
                    )
                    user_span.set_attribute("user_transcript", turn.user_transcript)
                    user_ctx = trace.set_span_in_context(user_span)

                    user_cursor_ns = _emit_component_span(
                        _tracer,
                        name="VADMetrics",
                        context=user_ctx,
                        start_ns=user_cursor_ns,
                        duration_ms=vals["vad_metrics_duration_ms"],
                        attributes=_merge_component_attributes(
                            turn.vad_attributes,
                            {
                                "eou_delay_ms": vals["vad_duration_ms"],
                            },
                        ),
                    )
                    user_cursor_ns = _emit_component_span(
                        _tracer,
                        name="STTMetrics",
                        context=user_ctx,
                        start_ns=user_cursor_ns,
                        duration_ms=vals["stt_span_duration_ms"],
                        attributes={
                            **turn.stt_attributes,
                            "user_transcript": turn.user_transcript,
                            "stt_status": turn.stt_status,
                            "stt_processing_ms": vals["stt_processing_ms"],
                            "stt_finalization_ms": vals["stt_finalization_ms"],
                            "stt_total_latency_ms": vals["stt_total_latency_ms"],
                        },
                        observation_output=turn.user_transcript,
                    )
                    user_cursor_ns = _emit_component_span(
                        _tracer,
                        name="EOUMetrics",
                        context=user_ctx,
                        start_ns=user_cursor_ns,
                        duration_ms=vals["vad_duration_ms"],
                        attributes=_merge_component_attributes(
                            turn.eou_attributes,
                            {
                                "end_of_utterance_delay_ms": vals["vad_duration_ms"],
                                "transcription_delay_ms": vals["stt_finalization_ms"],
                                "on_user_turn_completed_delay_ms": vals[
                                    "eou_on_user_turn_completed_ms"
                                ],
                            },
                        ),
                        observation_output=str(vals["vad_duration_ms"]),
                    )
                finally:
                    _close_container_span(
                        user_span,
                        start_ns=user_start_ns,
                        end_ns=user_cursor_ns,
                    )
                cursor_ns = user_cursor_ns

                for block in phase_blocks:
                    if isinstance(block, ResponsePhaseBlock):
                        phase_start_ns = cursor_ns
                        phase_span = _tracer.start_span(
                            f"agent_response_phase_{block.index}",
                            context=ctx,
                            start_time=phase_start_ns,
                        )
                        phase_cursor_ns = phase_start_ns
                        try:
                            phase_kind = "single"
                            if vals["tool_execution_count"] > 0 and block.index == 1:
                                phase_kind = "pre-tool"
                            elif vals["tool_execution_count"] > 0 and block.index > 1:
                                phase_kind = "post-tool"
                            phase_span.set_attribute("phase.index", block.index)
                            phase_span.set_attribute("phase.kind", phase_kind)
                            phase_ctx = trace.set_span_in_context(phase_span)
                            phase_text = _phase_response_text(block)
                            if (
                                not phase_text
                                and last_response_phase_index is not None
                                and block.index == last_response_phase_index
                            ):
                                phase_text = _latest_assistant_text(turn)

                            for llm_idx, llm_call in enumerate(
                                block.llm_calls, start=1
                            ):
                                llm_visible_latency_ms = _preferred_visible_latency_ms(
                                    llm_call.ttft_ms,
                                    llm_call.duration_ms,
                                )
                                phase_cursor_ns = _emit_component_span(
                                    _tracer,
                                    name="LLMMetrics",
                                    context=phase_ctx,
                                    start_ns=phase_cursor_ns,
                                    duration_ms=llm_visible_latency_ms,
                                    advance_ms=llm_call.duration_ms,
                                    attributes=_merge_component_attributes(
                                        llm_call.attributes,
                                        {
                                            "prompt_text": turn.prompt_text,
                                            "response_text": phase_text,
                                            "ttft_ms": llm_call.ttft_ms,
                                            "llm_total_latency_ms": llm_call.duration_ms,
                                            "total_duration_ms": llm_call.duration_ms,
                                            "phase_index": block.index,
                                            "phase_call_index": llm_idx,
                                        },
                                    ),
                                    observation_input=turn.prompt_text,
                                    observation_output=phase_text,
                                )

                            for tts_idx, tts_call in enumerate(
                                block.tts_calls, start=1
                            ):
                                spoken_text = tts_call.assistant_text or phase_text
                                tts_visible_latency_ms = _preferred_visible_latency_ms(
                                    tts_call.ttfb_ms,
                                    tts_call.duration_ms,
                                )
                                phase_cursor_ns = _emit_component_span(
                                    _tracer,
                                    name="TTSMetrics",
                                    context=phase_ctx,
                                    start_ns=phase_cursor_ns,
                                    duration_ms=tts_visible_latency_ms,
                                    advance_ms=tts_call.duration_ms,
                                    attributes=_merge_component_attributes(
                                        tts_call.attributes,
                                        {
                                            "assistant_text": spoken_text,
                                            "assistant_text_missing": turn.assistant_text_missing,
                                            "ttfb_ms": tts_call.ttfb_ms,
                                            "tts_total_latency_ms": tts_call.duration_ms,
                                            "total_duration_ms": tts_call.duration_ms,
                                            "phase_index": block.index,
                                            "phase_call_index": tts_idx,
                                        },
                                    ),
                                    observation_input=spoken_text,
                                    observation_output=spoken_text,
                                )
                        finally:
                            _close_container_span(
                                phase_span,
                                start_ns=phase_start_ns,
                                end_ns=phase_cursor_ns,
                            )
                        cursor_ns = phase_cursor_ns
                        continue

                    tool_start_ns = cursor_ns
                    tool_span = _tracer.start_span(
                        f"tool_execution_{block.index}",
                        context=ctx,
                        start_time=tool_start_ns,
                    )
                    tool_cursor_ns = tool_start_ns
                    try:
                        tool_span.set_attribute("tool.execution.index", block.index)
                        tool_span.set_attribute(
                            "tool.execution.call_count", len(block.execution.tool_calls)
                        )
                        tool_span.set_attribute(
                            "tool.execution.error_count",
                            sum(1 for call in block.execution.tool_calls if call.is_error),
                        )
                        tool_ctx = trace.set_span_in_context(tool_span)
                        for tool_call in block.execution.tool_calls:
                            tool_cursor_ns = _emit_component_span(
                                _tracer,
                                name="ToolCall",
                                context=tool_ctx,
                                start_ns=tool_cursor_ns,
                                duration_ms=tool_call.duration_ms,
                                attributes={
                                    "tool.name": tool_call.name,
                                    "tool.call_id": tool_call.call_id,
                                    "tool.is_error": tool_call.is_error,
                                    "tool.created_at": tool_call.created_at,
                                    "tool.completed_at": tool_call.completed_at,
                                },
                                observation_input=tool_call.arguments,
                                observation_output=tool_call.output,
                            )
                    finally:
                        _close_container_span(
                            tool_span,
                            start_ns=tool_start_ns,
                            end_ns=tool_cursor_ns,
                        )
                    cursor_ns = tool_cursor_ns

                summary_start_ns = cursor_ns
                summary_span = _tracer.start_span(
                    "metrics_summary",
                    context=ctx,
                    start_time=summary_start_ns,
                )
                summary_cursor_ns = summary_start_ns
                try:
                    summary_ctx = trace.set_span_in_context(summary_span)
                    first_latency_ms = vals["perceived_latency_first_audio_ms"]
                    if first_latency_ms is not None:
                        summary_cursor_ns = _emit_component_span(
                            _tracer,
                            name="perceived_latency_first_audio",
                            context=summary_ctx,
                            start_ns=summary_cursor_ns,
                            duration_ms=first_latency_ms,
                            attributes={
                                "speech_end_to_assistant_speech_start_ms": first_latency_ms,
                                "eou_delay_ms": vals["vad_duration_ms"],
                                "llm_ttft_ms": vals["llm_ttft_ms"],
                                "tool_calls_total_ms": vals["tool_calls_total_ms"],
                                "llm_to_tts_handoff_ms": vals["llm_to_tts_handoff_ms"],
                                "tts_ttfb_ms": vals["tts_ttfb_ms"],
                            },
                            observation_output=str(first_latency_ms),
                        )
                    second_latency_ms = vals["perceived_latency_second_audio_ms"]
                    if second_latency_ms is not None:
                        summary_cursor_ns = _emit_component_span(
                            _tracer,
                            name="perceived_latency_second_audio",
                            context=summary_ctx,
                            start_ns=summary_cursor_ns,
                            duration_ms=second_latency_ms,
                            attributes={
                                "tool_to_audio_wait_ms": second_latency_ms,
                                "tool_calls_total_ms": vals["tool_calls_total_ms"],
                            },
                            observation_output=str(second_latency_ms),
                        )
                finally:
                    _close_container_span(
                        summary_span,
                        start_ns=summary_start_ns,
                        end_ns=summary_cursor_ns,
                    )
                cursor_ns = summary_cursor_ns
            finally:
                total_ns = _ms_to_ns(_total_duration_ms(turn))
                _close_span_at(
                    turn_span, max(cursor_ns, root_start_ns + total_ns)
                )

            logger.info(
                "Langfuse turn trace emitted: trace_id=%s turn_id=%s session_id=%s room_id=%s participant_id=%s finalization_reason=%s assistant_text_source=%s emit_wait_ms=%.1f",
                turn.trace_id,
                turn.turn_id,
                turn.session_id,
                turn.room_id,
                turn.participant_id,
                turn.finalization_reason,
                turn.assistant_text_source,
                max((time() - turn.emit_ready_at) * 1000.0, 0.0)
                if turn.emit_ready_at is not None
                else 0.0,
            )
            asyncio.create_task(self._flush_tracer_provider())
        except Exception as exc:
            logger.error(f"Failed to create Langfuse turn trace: {exc}")

    async def _flush_tracer_provider(self) -> None:
        try:
            tracer_provider = trace.get_tracer_provider()
            force_flush = getattr(tracer_provider, "force_flush", None)
            if not callable(force_flush):
                return
            await asyncio.wait_for(
                asyncio.to_thread(force_flush),
                timeout=self._trace_flush_timeout_sec,
            )
        except TimeoutError:
            logger.debug("Langfuse tracer force_flush timed out")
        except Exception as exc:
            logger.debug(f"Failed to force flush tracer provider: {exc}")


# ------------------------------------------------------------------
# Pure helpers (module-level)
# ------------------------------------------------------------------


def _normalize_optional_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _sum_llm_duration_ms(calls: list[LLMCallTrace]) -> float:
    total = 0.0
    for call in calls:
        if call.duration_ms is None:
            continue
        total += max(call.duration_ms, 0.0)
    return total


def _sum_tts_duration_ms(calls: list[TTSCallTrace]) -> float:
    total = 0.0
    for call in calls:
        if call.duration_ms is None:
            continue
        total += max(call.duration_ms, 0.0)
    return total


def _flatten_tool_calls(executions: list[ToolExecutionTrace]) -> list[ToolCallTrace]:
    calls: list[ToolCallTrace] = []
    for execution in executions:
        calls.extend(execution.tool_calls)
    return calls


def _latest_llm_call_after_order(
    calls: list[LLMCallTrace],
    *,
    min_order: int,
    max_order: Optional[int] = None,
) -> Optional[LLMCallTrace]:
    matched: Optional[LLMCallTrace] = None
    for call in calls:
        if call.observed_order <= min_order:
            continue
        if max_order is not None and call.observed_order > max_order:
            continue
        matched = call
    return matched


def _estimate_tts_first_audio_at(
    *,
    metric_attributes: dict[str, Any],
    duration_sec: float,
    ttfb_sec: float,
) -> Optional[float]:
    timestamp_sec = _to_optional_float(metric_attributes.get("timestamp"))
    duration_attr_sec = _to_optional_float(metric_attributes.get("duration"))
    ttfb_attr_sec = _to_optional_float(metric_attributes.get("ttfb"))
    if timestamp_sec is None:
        return None
    resolved_duration_sec = (
        duration_attr_sec
        if duration_attr_sec is not None and duration_attr_sec > 0
        else max(duration_sec, 0.0)
    )
    resolved_ttfb_sec = (
        ttfb_attr_sec
        if ttfb_attr_sec is not None and ttfb_attr_sec >= 0
        else max(ttfb_sec, 0.0)
    )
    first_audio_at = timestamp_sec - max(resolved_duration_sec - resolved_ttfb_sec, 0.0)
    return _resolved_event_timestamp(first_audio_at)


def _build_phase_blocks(
    turn: TraceTurn,
) -> list[ResponsePhaseBlock | ToolExecutionBlock]:
    timeline = sorted(turn.timeline_events, key=lambda event: event.order)
    blocks: list[ResponsePhaseBlock | ToolExecutionBlock] = []
    phase_index = 1
    current_phase = ResponsePhaseBlock(index=phase_index)

    def _flush_phase() -> None:
        if current_phase.llm_calls or current_phase.tts_calls:
            blocks.append(
                ResponsePhaseBlock(
                    index=current_phase.index,
                    llm_calls=list(current_phase.llm_calls),
                    tts_calls=list(current_phase.tts_calls),
                )
            )

    for event in timeline:
        if event.kind == "llm":
            if 0 <= event.ref_index < len(turn.llm_calls):
                current_phase.llm_calls.append(turn.llm_calls[event.ref_index])
            continue
        if event.kind == "tts":
            if 0 <= event.ref_index < len(turn.tts_calls):
                current_phase.tts_calls.append(turn.tts_calls[event.ref_index])
            continue
        if event.kind == "tool_execution":
            _flush_phase()
            if 0 <= event.ref_index < len(turn.tool_executions):
                blocks.append(
                    ToolExecutionBlock(
                        index=event.ref_index + 1,
                        execution=turn.tool_executions[event.ref_index],
                    )
                )
            phase_index += 1
            current_phase = ResponsePhaseBlock(index=phase_index)
            continue

    _flush_phase()

    if blocks:
        return blocks

    fallback_phase = ResponsePhaseBlock(index=1)
    fallback_phase.llm_calls.extend(turn.llm_calls)
    fallback_phase.tts_calls.extend(turn.tts_calls)
    if fallback_phase.llm_calls or fallback_phase.tts_calls:
        return [fallback_phase]
    return []


def _phase_response_text(
    block: ResponsePhaseBlock,
) -> str:
    for tts_call in reversed(block.tts_calls):
        if tts_call.assistant_text.strip():
            return tts_call.assistant_text.strip()
    return ""


def _last_response_phase_index(
    phase_blocks: list[ResponsePhaseBlock | ToolExecutionBlock],
) -> Optional[int]:
    last_index: Optional[int] = None
    for block in phase_blocks:
        if isinstance(block, ResponsePhaseBlock):
            last_index = block.index
    return last_index


def _latest_assistant_text(turn: TraceTurn) -> str:
    return (turn.assistant_text or turn.response_text).strip()


def _phase_latest_observed_order(block: ResponsePhaseBlock) -> Optional[int]:
    orders = [call.observed_order for call in block.tts_calls]
    if orders:
        return max(orders)
    llm_orders = [call.observed_order for call in block.llm_calls]
    if llm_orders:
        return max(llm_orders)
    return None


def _assistant_text_newer_than_phase(
    *,
    turn: TraceTurn,
    block: ResponsePhaseBlock,
) -> bool:
    latest_assistant = _latest_assistant_text(turn)
    if not latest_assistant:
        return False
    phase_latest_order = _phase_latest_observed_order(block)
    if phase_latest_order is None:
        return False
    return bool(
        turn.assistant_text_updated_order is not None
        and turn.assistant_text_updated_order > phase_latest_order
    )


def _ordered_phase_response_texts(
    *,
    turn: TraceTurn,
    phase_blocks: list[ResponsePhaseBlock | ToolExecutionBlock],
) -> list[str]:
    response_texts: list[str] = []
    last_response_phase_index = _last_response_phase_index(phase_blocks)
    latest_assistant = _latest_assistant_text(turn)
    for block in phase_blocks:
        if not isinstance(block, ResponsePhaseBlock):
            continue
        phase_text = _phase_response_text(block)
        is_last_phase = (
            last_response_phase_index is not None
            and block.index == last_response_phase_index
        )
        if is_last_phase and _assistant_text_newer_than_phase(turn=turn, block=block):
            phase_text = latest_assistant
        elif not phase_text and is_last_phase:
            phase_text = latest_assistant
        if phase_text:
            if response_texts and response_texts[-1] == phase_text:
                continue
            response_texts.append(phase_text)
    return response_texts


def _build_trace_output(
    *,
    turn: TraceTurn,
    phase_blocks: list[ResponsePhaseBlock | ToolExecutionBlock],
) -> str:
    if turn.tool_post_response_missing and not _has_post_tool_assistant_text(turn):
        fallback = _latest_assistant_text(turn)
        if fallback:
            return fallback

    phase_texts = _ordered_phase_response_texts(
        turn=turn,
        phase_blocks=phase_blocks,
    )
    if phase_texts:
        return "\n".join(phase_texts)
    return _latest_assistant_text(turn)


def _backfill_next_missing_tts_assistant_text(
    turn: TraceTurn,
    assistant_text: str,
) -> None:
    normalized = assistant_text.strip()
    if not normalized:
        return
    for tts_call in turn.tts_calls:
        if tts_call.assistant_text.strip():
            continue
        tts_call.assistant_text = normalized
        return


def _reconcile_assistant_text_with_tts_calls(
    *,
    turn: TraceTurn,
    assistant_text: str,
    previous_assistant_text: str,
) -> None:
    normalized = assistant_text.strip()
    if not normalized:
        return
    previous = previous_assistant_text.strip()
    post_tool_min_order = turn.last_tool_event_order
    for tts_call in reversed(turn.tts_calls):
        if (
            post_tool_min_order is not None
            and tts_call.observed_order <= post_tool_min_order
        ):
            continue
        existing = tts_call.assistant_text.strip()
        if not existing or (previous and existing == previous):
            tts_call.assistant_text = normalized
        return
    _backfill_next_missing_tts_assistant_text(turn, normalized)


def _has_post_tool_assistant_text(turn: TraceTurn) -> bool:
    if turn.last_tool_event_order is None:
        return False
    if not _latest_assistant_text(turn):
        return False
    if (
        turn.last_tool_event_at is not None
        and turn.assistant_text_updated_at is not None
    ):
        if turn.assistant_text_updated_at > turn.last_tool_event_at:
            return True
        if turn.assistant_text_updated_at < turn.last_tool_event_at:
            return False
    return bool(
        turn.assistant_text_updated_order is not None
        and turn.assistant_text_updated_order > turn.last_tool_event_order
    )


def _tool_error_fallback_text(turn: TraceTurn) -> str:
    has_tool_error = any(
        tool_call.is_error for tool_call in _flatten_tool_calls(turn.tool_executions)
    )
    if not has_tool_error:
        return ""
    return _TOOL_ERROR_FALLBACK_TEXT


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


def _resolved_event_timestamp(event_created_at: Optional[float]) -> float:
    now = time()
    if event_created_at is None:
        return now
    # Some test fixtures provide synthetic timestamps that are not wall-clock.
    if abs(now - event_created_at) > 30 * 24 * 60 * 60:
        return now
    return event_created_at


def _build_tool_call_trace(
    *,
    function_call: Any,
    function_call_output: Any,
    event_created_at: Optional[float],
) -> ToolCallTrace:
    name = _stringify_observation(getattr(function_call, "name", None))
    call_id = _stringify_observation(getattr(function_call, "call_id", None))
    arguments = _stringify_observation(getattr(function_call, "arguments", None))

    output_text = ""
    is_error = False
    completed_at = event_created_at

    if function_call_output is not None:
        output_text = _stringify_observation(getattr(function_call_output, "output", None))
        is_error = bool(getattr(function_call_output, "is_error", False))
        completed_at = _to_optional_float(getattr(function_call_output, "created_at", None)) or event_created_at

    created_at = _to_optional_float(getattr(function_call, "created_at", None)) or event_created_at
    duration_ms: Optional[float] = None
    if created_at is not None and completed_at is not None:
        duration_ms = max((completed_at - created_at) * 1000.0, 0.0)

    return ToolCallTrace(
        name=name,
        call_id=call_id,
        arguments=arguments,
        output=output_text,
        is_error=is_error,
        created_at=created_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
    )


def _duration_to_ms(duration: float, fallback: float) -> float:
    chosen = duration if duration > 0 else fallback
    return max(chosen, 0.0) * 1000.0


def _duration_to_ms_or_none(duration: float, fallback: float) -> Optional[float]:
    chosen = duration if duration > 0 else fallback
    if chosen <= 0:
        return None
    return chosen * 1000.0


def _ms_to_ns(ms: float) -> int:
    return int(max(ms, 0.0) * 1_000_000)


def _recompute_perceived_first_audio_latency(turn: TraceTurn) -> None:
    first_llm = turn.llm_calls[0] if turn.llm_calls else None
    first_tts = turn.tts_calls[0] if turn.tts_calls else None
    turn.llm_duration_ms = first_llm.duration_ms if first_llm else turn.llm_duration_ms
    turn.llm_ttft_ms = first_llm.ttft_ms if first_llm else turn.llm_ttft_ms
    turn.tts_duration_ms = first_tts.duration_ms if first_tts else turn.tts_duration_ms
    turn.tts_ttfb_ms = first_tts.ttfb_ms if first_tts else turn.tts_ttfb_ms
    turn.llm_total_latency_ms = _sum_llm_duration_ms(turn.llm_calls)
    turn.tts_total_latency_ms = _sum_tts_duration_ms(turn.tts_calls)
    turn.perceived_latency_first_audio_ms = _compute_conversational_latency_ms(
        vad_duration_ms=turn.vad_duration_ms,
        llm_ttft_ms=turn.llm_ttft_ms,
        tts_ttfb_ms=turn.tts_ttfb_ms,
    )
    turn.conversational_latency_ms = turn.perceived_latency_first_audio_ms


def _compute_conversational_latency_ms(
    *,
    vad_duration_ms: Optional[float],
    llm_ttft_ms: Optional[float],
    tts_ttfb_ms: Optional[float],
) -> Optional[float]:
    components = (vad_duration_ms, llm_ttft_ms, tts_ttfb_ms)
    if any(c is None for c in components):
        return None
    return sum(c for c in components if c is not None)


def _compute_llm_to_tts_handoff_ms(
    *,
    total_latency_ms: Optional[float],
    vad_duration_ms: Optional[float],
    llm_ttft_ms: Optional[float],
    tts_ttfb_ms: Optional[float],
) -> Optional[float]:
    if total_latency_ms is None:
        return None
    baseline = _compute_conversational_latency_ms(
        vad_duration_ms=vad_duration_ms,
        llm_ttft_ms=llm_ttft_ms,
        tts_ttfb_ms=tts_ttfb_ms,
    )
    if baseline is None:
        return None
    return max(total_latency_ms - baseline, 0.0)


def _total_duration_ms(turn: TraceTurn) -> float:
    all_tool_calls = _flatten_tool_calls(turn.tool_executions)
    tool_calls_total = _tool_calls_total_duration_ms(all_tool_calls)
    llm = _sum_llm_duration_ms(turn.llm_calls)
    tts = _sum_tts_duration_ms(turn.tts_calls)
    handoff = max(turn.llm_to_tts_handoff_ms or 0.0, 0.0)
    calculated = (
        (turn.vad_duration_ms or 0.0)
        + llm
        + tool_calls_total
        + handoff
        + tts
    )
    if turn.perceived_latency_first_audio_ms is not None:
        calculated = max(calculated, turn.perceived_latency_first_audio_ms)
    if turn.perceived_latency_second_audio_ms is not None:
        calculated = max(calculated, turn.perceived_latency_second_audio_ms)
    return calculated


def _build_turn_pipeline_summary(
    turn: TraceTurn,
    *,
    partial: bool,
) -> dict[str, Any]:
    phase_blocks = _build_phase_blocks(turn)
    response_blocks = [
        block for block in phase_blocks if isinstance(block, ResponsePhaseBlock)
    ]
    pre_tool_phase = response_blocks[0] if response_blocks else None

    all_tool_calls = _flatten_tool_calls(turn.tool_executions)
    has_tools = bool(turn.tool_executions or turn.tool_step_announced or all_tool_calls)

    post_tool_phase: Optional[ResponsePhaseBlock] = None
    if has_tools:
        for block in reversed(response_blocks):
            if block.index > 1:
                post_tool_phase = block
                break

    tool_payloads: list[dict[str, Any]] = []
    tool_total_ms = 0.0
    for tool_call in all_tool_calls:
        duration_ms = max(tool_call.duration_ms or 0.0, 0.0)
        tool_total_ms += duration_ms
        tool_payloads.append(
            {
                "name": tool_call.name or "tool_call",
                "call_id": tool_call.call_id,
                "duration_seconds": duration_ms / 1000.0,
                "duration_ms": duration_ms,
                "is_error": tool_call.is_error,
            }
        )

    first_audio_ms = (
        max(turn.perceived_latency_first_audio_ms, 0.0)
        if turn.perceived_latency_first_audio_ms is not None
        else None
    )
    second_audio_ms = (
        max(turn.perceived_latency_second_audio_ms, 0.0)
        if turn.perceived_latency_second_audio_ms is not None
        else None
    )
    total_turn_ms = _total_duration_ms(turn)

    payload: dict[str, Any] = {
        "type": "turn_pipeline_summary",
        "timestamp": time(),
        "turn_id": turn.turn_id,
        "session_id": turn.session_id,
        "partial": partial,
        "has_tools": has_tools,
        "phases": [
            _pipeline_phase_payload(
                phase_id=1,
                label="Turn Detection",
                sublabel="EOU Delay",
                duration_ms=max(turn.vad_duration_ms or 0.0, 0.0),
            ),
            _pipeline_phase_payload(
                phase_id=2,
                label="Thinking",
                sublabel="LLM TTFT",
                duration_ms=_first_llm_ttft_ms(pre_tool_phase),
            ),
            _pipeline_phase_payload(
                phase_id=3,
                label="Voice Generation",
                sublabel="TTS TTFB",
                duration_ms=_first_tts_ttfb_ms(pre_tool_phase),
            ),
        ],
        "tool_phase": None,
        "post_tool_phases": [],
        "first_audio_latency_seconds": (
            first_audio_ms / 1000.0 if first_audio_ms is not None else None
        ),
        "first_audio_latency_ms": first_audio_ms,
        "second_audio_latency_seconds": (
            second_audio_ms / 1000.0 if second_audio_ms is not None else None
        ),
        "second_audio_latency_ms": second_audio_ms,
        "total_turn_duration_seconds": total_turn_ms / 1000.0,
        "total_turn_duration_ms": total_turn_ms,
    }

    if has_tools:
        payload["tool_phase"] = {
            "execution_count": len(turn.tool_executions),
            "tools": tool_payloads,
            "total_duration_seconds": tool_total_ms / 1000.0,
            "total_duration_ms": tool_total_ms,
        }
        payload["post_tool_phases"] = [
            _pipeline_phase_payload(
                phase_id=5,
                label="Thinking",
                sublabel="LLM TTFT",
                duration_ms=_first_llm_ttft_ms(post_tool_phase),
            ),
            _pipeline_phase_payload(
                phase_id=6,
                label="Voice Generation",
                sublabel="TTS TTFB",
                duration_ms=_first_tts_ttfb_ms(post_tool_phase),
            ),
        ]

    return payload


def _pipeline_phase_payload(
    *,
    phase_id: int,
    label: str,
    sublabel: str,
    duration_ms: Optional[float],
) -> dict[str, Any]:
    resolved_ms = max(duration_ms, 0.0) if duration_ms is not None else None
    return {
        "id": phase_id,
        "label": label,
        "sublabel": sublabel,
        "duration_seconds": (resolved_ms / 1000.0) if resolved_ms is not None else None,
        "duration_ms": resolved_ms,
    }


def _first_llm_ttft_ms(phase: Optional[ResponsePhaseBlock]) -> Optional[float]:
    if phase is None:
        return None
    for call in phase.llm_calls:
        if call.ttft_ms is None:
            continue
        return max(call.ttft_ms, 0.0)
    return None


def _first_tts_ttfb_ms(phase: Optional[ResponsePhaseBlock]) -> Optional[float]:
    if phase is None:
        return None
    for call in phase.tts_calls:
        if call.ttfb_ms is None:
            continue
        return max(call.ttfb_ms, 0.0)
    return None


def _prepare_span_values(turn: TraceTurn) -> dict[str, Any]:
    """Pre-compute derived values used by span emission."""
    user_input_duration_ms = 0.0 if turn.user_transcript else None
    vad_duration_ms = max(turn.vad_duration_ms or 0.0, 0.0)
    vad_metrics_duration_ms = _duration_attribute_to_ms(
        turn.vad_attributes.get("inference_duration_total")
    )
    stt_processing_ms = (
        max(turn.stt_duration_ms, 0.0) if turn.stt_duration_ms is not None else None
    )
    stt_finalization_ms = (
        max(turn.stt_finalization_ms, 0.0)
        if turn.stt_finalization_ms is not None
        else None
    )
    stt_total_latency_ms = (
        max(turn.stt_total_latency_ms, 0.0)
        if turn.stt_total_latency_ms is not None
        else None
    )
    stt_span_duration_ms: Optional[float] = None
    if stt_processing_ms is not None and stt_processing_ms > 0:
        stt_span_duration_ms = stt_processing_ms
    elif stt_finalization_ms is not None and stt_finalization_ms > 0:
        stt_span_duration_ms = stt_finalization_ms
    else:
        stt_span_duration_ms = stt_total_latency_ms

    llm_duration_ms = max(turn.llm_duration_ms or 0.0, 0.0)
    llm_ttft_ms = max(turn.llm_ttft_ms or 0.0, 0.0)
    llm_total_latency_ms = max(_sum_llm_duration_ms(turn.llm_calls), 0.0)
    tts_duration_ms = max(turn.tts_duration_ms or 0.0, 0.0)
    tts_ttfb_ms = max(turn.tts_ttfb_ms or 0.0, 0.0)
    tts_total_latency_ms = max(_sum_tts_duration_ms(turn.tts_calls), 0.0)
    perceived_latency_first_audio_ms = (
        max(turn.perceived_latency_first_audio_ms, 0.0)
        if turn.perceived_latency_first_audio_ms is not None
        else None
    )
    perceived_latency_second_audio_ms = (
        max(turn.perceived_latency_second_audio_ms, 0.0)
        if turn.perceived_latency_second_audio_ms is not None
        else None
    )
    all_tool_calls = _flatten_tool_calls(turn.tool_executions)
    tool_calls_total_ms = _tool_calls_total_duration_ms(all_tool_calls)
    llm_to_tts_handoff_ms = (
        max(turn.llm_to_tts_handoff_ms or 0.0, 0.0)
        if turn.llm_to_tts_handoff_ms is not None
        else None
    )
    tool_call_count = len(all_tool_calls)
    tool_error_count = sum(1 for call in all_tool_calls if call.is_error)
    tool_execution_count = len(turn.tool_executions)
    eou_on_user_turn_completed_ms = (
        max(turn.eou_on_user_turn_completed_ms, 0.0)
        if turn.eou_on_user_turn_completed_ms is not None
        else None
    )
    return {
        "user_input_duration_ms": user_input_duration_ms,
        "vad_duration_ms": vad_duration_ms,
        "vad_metrics_duration_ms": vad_metrics_duration_ms,
        "stt_processing_ms": stt_processing_ms,
        "stt_finalization_ms": stt_finalization_ms,
        "stt_total_latency_ms": stt_total_latency_ms,
        "stt_span_duration_ms": stt_span_duration_ms,
        "eou_on_user_turn_completed_ms": eou_on_user_turn_completed_ms,
        "llm_duration_ms": llm_duration_ms,
        "llm_ttft_ms": llm_ttft_ms,
        "llm_total_latency_ms": llm_total_latency_ms,
        "tts_duration_ms": tts_duration_ms,
        "tts_total_latency_ms": tts_total_latency_ms,
        "tts_ttfb_ms": tts_ttfb_ms,
        "conversational_latency_ms": perceived_latency_first_audio_ms,
        "perceived_latency_first_audio_ms": perceived_latency_first_audio_ms,
        "perceived_latency_second_audio_ms": perceived_latency_second_audio_ms,
        "tool_calls_total_ms": tool_calls_total_ms,
        "tool_call_count": tool_call_count,
        "tool_error_count": tool_error_count,
        "tool_execution_count": tool_execution_count,
        "llm_to_tts_handoff_ms": llm_to_tts_handoff_ms,
    }


def _set_root_attributes(
    span: Any,
    turn: TraceTurn,
    vals: dict[str, Any],
    trace_output: str,
) -> None:
    """Set all attributes on the root turn span."""
    attrs: dict[str, Any] = {
        "session_id": turn.session_id,
        "room_id": turn.room_id,
        "participant_id": turn.participant_id,
        "turn_id": turn.turn_id,
        "langfuse.session.id": turn.session_id,
        "session.id": turn.session_id,
        "langfuse.user.id": turn.participant_id,
        "user.id": turn.participant_id,
        "langfuse.trace.name": "turn",
        "langfuse.trace.input": turn.user_transcript,
        "langfuse.trace.output": trace_output,
        "langfuse.trace.public": bool(settings.langfuse.LANGFUSE_PUBLIC_TRACES),
        "langfuse.trace.metadata.room_id": turn.room_id,
        "langfuse.trace.metadata.participant_id": turn.participant_id,
        "langfuse.trace.metadata.turn_id": turn.turn_id,
        "langfuse.trace.metadata.assistant_text_missing": turn.assistant_text_missing,
        "langfuse.trace.metadata.assistant_text_source": turn.assistant_text_source,
        "langfuse.trace.metadata.stt_status": turn.stt_status,
        "langfuse.trace.metadata.tool_phase_announced": turn.tool_step_announced,
        "langfuse.trace.metadata.tool_post_response_missing": turn.tool_post_response_missing,
        "langfuse.trace.metadata.user_turn_committed": turn.user_turn_committed,
        "langfuse.trace.metadata.assistant_audio_started": turn.assistant_audio_started,
        "langfuse.trace.metadata.interrupted": turn.interrupted,
        "langfuse.trace.metadata.interrupted_reason": turn.interrupted_reason,
        "langfuse.trace.metadata.finalization_reason": turn.finalization_reason,
        "langfuse.trace.metadata.emit_ready_at": turn.emit_ready_at,
        "langfuse.trace.metadata.coalesced_turn_count": len(turn.coalesced_turn_ids),
        "langfuse.trace.metadata.coalesced_fragment_count": turn.coalesced_fragment_count,
        "langfuse.trace.metadata.coalesced_turn_ids": turn.coalesced_turn_ids,
        "langfuse.trace.metadata.coalesced_inputs": turn.coalesced_user_transcripts,
        "duration_ms": _total_duration_ms(turn),
        "latency_ms.user_input": vals["user_input_duration_ms"],
        "latency_ms.vad": vals["vad_duration_ms"],
        "latency_ms.eou_delay": vals["vad_duration_ms"],
        "latency_ms.stt": vals["stt_span_duration_ms"],
        "latency_ms.stt_processing": vals["stt_processing_ms"],
        "latency_ms.stt_finalization": vals["stt_finalization_ms"],
        "latency_ms.stt_total": vals["stt_total_latency_ms"],
        "latency_ms.eou_on_user_turn_completed": vals[
            "eou_on_user_turn_completed_ms"
        ],
        "latency_ms.llm": vals["llm_duration_ms"],
        "latency_ms.llm_ttft": vals["llm_ttft_ms"],
        "latency_ms.llm_total": vals["llm_total_latency_ms"],
        "latency_ms.tts": vals["tts_duration_ms"],
        "latency_ms.tts_total": vals["tts_total_latency_ms"],
        "latency_ms.tts_ttfb": vals["tts_ttfb_ms"],
        "latency_ms.tool_calls_total": vals["tool_calls_total_ms"],
        "latency_ms.llm_to_tts_handoff": vals["llm_to_tts_handoff_ms"],
        "latency_ms.perceived_first_audio": vals["perceived_latency_first_audio_ms"],
        "latency_ms.perceived_second_audio": vals["perceived_latency_second_audio_ms"],
        "latency_ms.conversational": vals["perceived_latency_first_audio_ms"],
        "latency_ms.speech_end_to_assistant_speech_start": vals[
            "perceived_latency_first_audio_ms"
        ],
        "tool.call_count": vals["tool_call_count"],
        "tool.error_count": vals["tool_error_count"],
        "tool.execution_count": vals["tool_execution_count"],
        "tool.phase_announced": turn.tool_step_announced,
        "tool.post_response_missing": turn.tool_post_response_missing,
        "stt_status": turn.stt_status,
        "user_turn.committed": turn.user_turn_committed,
    }
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


def _sanitize_component_attributes(
    attributes: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if not attributes:
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        sanitized[key] = _safe_attribute_value(value)
    return sanitized


def _merge_component_attributes(
    existing: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in extra.items():
        if value is None:
            continue
        merged[key] = _safe_attribute_value(value)
    return merged


def _safe_attribute_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_attribute_value(v) for v in value]
    return str(value)


def _duration_attribute_to_ms(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return max(float(value), 0.0) * 1000.0
    return None


def _assistant_text_from_component_attributes(attributes: dict[str, Any]) -> str:
    for key in (
        "assistant_text",
        "spoken_text",
        "metadata.assistant_text",
        "metadata.spoken_text",
    ):
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _preferred_visible_latency_ms(
    preferred_ms: Optional[float],
    fallback_ms: Optional[float],
) -> Optional[float]:
    if preferred_ms is not None and preferred_ms >= 0.0:
        return preferred_ms
    if fallback_ms is not None and fallback_ms >= 0.0:
        return fallback_ms
    return None


def _tool_calls_total_duration_ms(tool_calls: list[ToolCallTrace]) -> float:
    total = 0.0
    for call in tool_calls:
        if call.duration_ms is None:
            continue
        total += max(call.duration_ms, 0.0)
    return total


def _stringify_observation(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


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


def _emit_component_span(
    _tracer: Any,
    *,
    name: str,
    context: Any,
    start_ns: int,
    duration_ms: Optional[float],
    advance_ms: Optional[float] = None,
    attributes: dict[str, Any],
    observation_input: Optional[str] = None,
    observation_output: Optional[str] = None,
) -> int:
    actual_ms = max(duration_ms, 0.0) if duration_ms is not None else None
    cursor_advance_ms = actual_ms
    if advance_ms is not None:
        cursor_advance_ms = max(advance_ms, 0.0)
        if actual_ms is not None:
            cursor_advance_ms = max(cursor_advance_ms, actual_ms)
    end_ns = start_ns + _ms_to_ns(actual_ms or 0.0)
    next_cursor_ns = start_ns + _ms_to_ns(cursor_advance_ms or 0.0)

    span = _tracer.start_span(name, context=context, start_time=start_ns)
    try:
        if actual_ms is not None:
            span.set_attribute("duration_ms", actual_ms)
        if observation_input is not None:
            span.set_attribute("input", observation_input)
            span.set_attribute("langfuse.observation.input", observation_input)
        if observation_output is not None:
            span.set_attribute("output", observation_output)
            span.set_attribute("langfuse.observation.output", observation_output)
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    finally:
        _close_span_at(span, end_ns)
    return max(end_ns, next_cursor_ns)


def _close_span_at(span: Any, end_ns: int) -> None:
    end = getattr(span, "end", None)
    if not callable(end):
        return
    try:
        end(end_time=end_ns)
    except TypeError:
        try:
            end(end_ns)
        except Exception:
            pass


def _set_observation_attributes(
    span: Any,
    *,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
) -> None:
    if input_text is not None:
        span.set_attribute("input", input_text)
        span.set_attribute("langfuse.observation.input", input_text)
    if output_text is not None:
        span.set_attribute("output", output_text)
        span.set_attribute("langfuse.observation.output", output_text)


def _close_container_span(span: Any, *, start_ns: int, end_ns: int) -> None:
    end = max(end_ns, start_ns)
    span.set_attribute("duration_ms", max((end - start_ns) / 1_000_000, 0.0))
    _close_span_at(span, end)
