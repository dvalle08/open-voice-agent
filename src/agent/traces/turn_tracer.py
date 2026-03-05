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
_DEFAULT_POST_TOOL_RESPONSE_TIMEOUT_MS = 30000.0
_DEFAULT_MAX_PENDING_TRACE_TASKS = 200
_DEFAULT_TRACE_FLUSH_TIMEOUT_SEC = 1.0
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
        self._trace_lock = asyncio.Lock()
        self._trace_emit_tasks: set[asyncio.Task[None]] = set()
        self._trace_finalize_tasks: dict[str, asyncio.Task[None]] = {}
        self._trace_finalize_task_versions: dict[str, int] = {}

        self._trace_finalize_timeout_sec = (
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
        async with self._trace_lock:
            self._pending_trace_turns.append(
                TraceTurn(
                    turn_id=str(uuid.uuid4()),
                    session_id=self._session_id,
                    room_id=room_id,
                    participant_id=self._participant_id,
                    user_transcript=user_transcript,
                    prompt_text=user_transcript,
                )
            )

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

            tts_attrs = _sanitize_component_attributes(metric_attributes)
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
            turn.tts_updated_at = time()
            turn.tts_updated_order = order

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
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._latest_turn_where(lambda c: bool(c.llm_calls))
            if not turn:
                turn = self._latest_turn_where(lambda _: True)
            if not turn:
                return None
            order = self._next_event_order(turn)
            turn.assistant_text = assistant_text
            turn.response_text = assistant_text
            turn.assistant_text_updated_at = time()
            turn.assistant_text_updated_order = order
            _backfill_next_missing_tts_assistant_text(turn, assistant_text)
            self._maybe_close_tool_phase(turn)
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

    def _select_turn_for_llm(self, speech_id: Optional[str]) -> Optional[TraceTurn]:
        if speech_id:
            matched = self._latest_turn_where(lambda c: c.speech_id == speech_id)
            if matched:
                return matched
            return self._next_turn_where(
                lambda c: c.speech_id is None and not c.llm_calls
            )
        return self._next_turn_where(lambda c: not c.llm_calls)

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
        base_complete = bool(
            turn.user_transcript
            and turn.assistant_text
            and turn.llm_calls
            and turn.tts_calls
        )
        if not base_complete:
            return False
        if turn.tool_phase_open:
            return False
        return not self._requires_post_tool_response(turn)

    def _should_schedule_finalize_timeout(self, turn: TraceTurn) -> bool:
        return bool(
            turn.llm_calls
            and turn.tts_calls
            and not self._is_complete(turn)
            and not (turn.tool_phase_open and turn.last_tool_event_at is None)
            and self._resolve_finalize_timeout_sec(turn) > 0.0
        )

    def _resolve_finalize_timeout_sec(self, turn: TraceTurn) -> float:
        if self._requires_post_tool_response(turn):
            return self._trace_post_tool_response_timeout_sec
        return self._trace_finalize_timeout_sec

    def _requires_post_tool_response(self, turn: TraceTurn) -> bool:
        if not turn.tool_step_announced and turn.last_tool_event_order is None:
            return False
        return not self._post_tool_response_observed(turn)

    def _post_tool_assistant_observed(self, turn: TraceTurn) -> bool:
        if turn.last_tool_event_order is None:
            return False
        return bool(
            turn.assistant_text_updated_order is not None
            and turn.assistant_text_updated_order > turn.last_tool_event_order
        )

    def _post_tool_response_observed(self, turn: TraceTurn) -> bool:
        if turn.last_tool_event_order is None:
            return False
        assistant_seen = self._post_tool_assistant_observed(turn)
        tts_seen = bool(
            turn.tts_updated_order is not None
            and turn.tts_updated_order > turn.last_tool_event_order
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
            fallback = self._best_available_assistant_text(
                turn,
                min_observed_order=(
                    turn.last_tool_event_order if tool_post_response_missing else None
                ),
                include_pending_agent_transcripts=not tool_post_response_missing,
            )
            if fallback:
                turn.assistant_text = fallback
                if not turn.response_text:
                    turn.response_text = fallback
            else:
                tool_error_fallback = ""
                if tool_post_response_missing:
                    tool_error_fallback = _tool_error_fallback_text(turn)

                if tool_error_fallback:
                    turn.assistant_text = tool_error_fallback
                    if not turn.response_text:
                        turn.response_text = tool_error_fallback
                else:
                    turn.assistant_text_missing = True
                    unavailable = "[assistant text unavailable]"
                    turn.assistant_text = unavailable
                    if not turn.response_text:
                        turn.response_text = unavailable

        turn.tool_phase_open = False
        if tool_post_response_missing:
            turn.tool_post_response_missing = True

        self._pending_trace_turns.remove(turn)
        self._cancel_finalize_timeout(turn.turn_id)
        return turn

    def _best_available_assistant_text(
        self,
        turn: TraceTurn,
        *,
        min_observed_order: Optional[int] = None,
        include_pending_agent_transcripts: bool = True,
    ) -> str:
        if turn.assistant_text.strip():
            if (
                min_observed_order is None
                or (
                    turn.assistant_text_updated_order is not None
                    and turn.assistant_text_updated_order > min_observed_order
                )
            ):
                return turn.assistant_text.strip()
        if turn.response_text.strip():
            if (
                min_observed_order is None
                or (
                    turn.assistant_text_updated_order is not None
                    and turn.assistant_text_updated_order > min_observed_order
                )
            ):
                return turn.response_text.strip()
        for tts_call in reversed(turn.tts_calls):
            if (
                min_observed_order is not None
                and tts_call.observed_order <= min_observed_order
            ):
                continue
            if tts_call.assistant_text.strip():
                return tts_call.assistant_text.strip()
        if include_pending_agent_transcripts and self._pending_agent_transcripts:
            return self._pending_agent_transcripts.popleft().strip()
        return ""

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
        await self._emit_turn_trace(turn)
        if turn.trace_id:
            await self._publisher.publish_trace_update(
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                trace_id=turn.trace_id,
            )

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
                                phase_cursor_ns = _emit_component_span(
                                    _tracer,
                                    name="LLMMetrics",
                                    context=phase_ctx,
                                    start_ns=phase_cursor_ns,
                                    duration_ms=llm_call.duration_ms,
                                    attributes=_merge_component_attributes(
                                        llm_call.attributes,
                                        {
                                            "prompt_text": turn.prompt_text,
                                            "response_text": phase_text,
                                            "ttft_ms": llm_call.ttft_ms,
                                            "llm_total_latency_ms": llm_call.duration_ms,
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
                                phase_cursor_ns = _emit_component_span(
                                    _tracer,
                                    name="TTSMetrics",
                                    context=phase_ctx,
                                    start_ns=phase_cursor_ns,
                                    duration_ms=tts_call.duration_ms,
                                    attributes=_merge_component_attributes(
                                        tts_call.attributes,
                                        {
                                            "assistant_text": spoken_text,
                                            "assistant_text_missing": turn.assistant_text_missing,
                                            "ttfb_ms": tts_call.ttfb_ms,
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
                "Langfuse turn trace emitted: trace_id=%s turn_id=%s session_id=%s room_id=%s participant_id=%s",
                turn.trace_id,
                turn.turn_id,
                turn.session_id,
                turn.room_id,
                turn.participant_id,
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


def _ordered_phase_response_texts(
    *,
    turn: TraceTurn,
    phase_blocks: list[ResponsePhaseBlock | ToolExecutionBlock],
) -> list[str]:
    response_texts: list[str] = []
    last_response_phase_index = _last_response_phase_index(phase_blocks)
    for block in phase_blocks:
        if not isinstance(block, ResponsePhaseBlock):
            continue
        phase_text = _phase_response_text(block)
        if (
            not phase_text
            and last_response_phase_index is not None
            and block.index == last_response_phase_index
        ):
            phase_text = _latest_assistant_text(turn)
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


def _has_post_tool_assistant_text(turn: TraceTurn) -> bool:
    if turn.last_tool_event_order is None:
        return False
    return bool(
        turn.assistant_text_updated_order is not None
        and turn.assistant_text_updated_order > turn.last_tool_event_order
        and _latest_assistant_text(turn)
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
        "langfuse.trace.metadata.room_id": turn.room_id,
        "langfuse.trace.metadata.participant_id": turn.participant_id,
        "langfuse.trace.metadata.turn_id": turn.turn_id,
        "langfuse.trace.metadata.assistant_text_missing": turn.assistant_text_missing,
        "langfuse.trace.metadata.stt_status": turn.stt_status,
        "langfuse.trace.metadata.tool_phase_announced": turn.tool_step_announced,
        "langfuse.trace.metadata.tool_post_response_missing": turn.tool_post_response_missing,
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


def _emit_component_span(
    _tracer: Any,
    *,
    name: str,
    context: Any,
    start_ns: int,
    duration_ms: Optional[float],
    attributes: dict[str, Any],
    observation_input: Optional[str] = None,
    observation_output: Optional[str] = None,
) -> int:
    actual_ms = max(duration_ms, 0.0) if duration_ms is not None else None
    end_ns = start_ns + _ms_to_ns(actual_ms or 0.0)

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
    return end_ns


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
