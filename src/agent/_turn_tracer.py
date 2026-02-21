"""Langfuse OTel turn tracer for LiveKit agent telemetry.

Creates one OpenTelemetry trace per finalized user turn and publishes
a ``trace_update`` message to the data channel when emission completes.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass
from time import time_ns
from typing import TYPE_CHECKING, Any, Callable, Optional

from opentelemetry import trace

from src.core.logger import logger
from src.core.settings import settings

if TYPE_CHECKING:
    from src.agent._channel_metrics import ChannelPublisher


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
    llm_duration_ms: Optional[float] = None
    llm_ttft_ms: Optional[float] = None
    llm_total_latency_ms: Optional[float] = None
    tts_duration_ms: Optional[float] = None
    tts_ttfb_ms: Optional[float] = None
    conversational_latency_ms: Optional[float] = None
    llm_to_tts_handoff_ms: Optional[float] = None
    trace_id: Optional[str] = None


_DEFAULT_TRACE_FINALIZE_TIMEOUT_MS = 8000.0
_DEFAULT_MAX_PENDING_TRACE_TASKS = 200
_DEFAULT_TRACE_FLUSH_TIMEOUT_SEC = 1.0


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
            _recompute_conversational_latency(turn)
            return turn

    async def attach_vad(
        self,
        *,
        duration: float,
        transcription_delay: float,
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
            if turn.stt_total_latency_ms > 0:
                turn.stt_status = "measured"
                if turn.stt_duration_ms is None:
                    turn.stt_duration_ms = turn.stt_total_latency_ms
            _recompute_conversational_latency(turn)
            return turn

    async def attach_llm(
        self,
        *,
        duration: float,
        ttft: float,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_where(
                lambda c: c.llm_duration_ms is None
            )
            if not turn:
                return None
            turn.prompt_text = turn.user_transcript
            turn.llm_duration_ms = _duration_to_ms(duration, 0.0)
            turn.llm_total_latency_ms = turn.llm_duration_ms
            turn.llm_ttft_ms = _duration_to_ms(ttft, 0.0)
            _recompute_conversational_latency(turn)
            return turn

    async def attach_tts(
        self,
        *,
        duration: float,
        fallback_duration: float,
        ttfb: float,
        observed_total_latency: Optional[float],
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_where(
                lambda c: (
                    c.llm_duration_ms is not None and c.tts_duration_ms is None
                )
            )
            if not turn:
                return None
            turn.tts_duration_ms = _duration_to_ms(duration, fallback_duration)
            turn.tts_ttfb_ms = _duration_to_ms(ttfb, 0.0)
            _recompute_conversational_latency(turn)
            if observed_total_latency is not None:
                observed_ms = observed_total_latency * 1000.0
                baseline_ms = turn.conversational_latency_ms
                turn.conversational_latency_ms = max(
                    observed_ms,
                    baseline_ms if baseline_ms is not None else 0.0,
                )
            turn.llm_to_tts_handoff_ms = _compute_llm_to_tts_handoff_ms(
                total_latency_ms=turn.conversational_latency_ms,
                vad_duration_ms=turn.vad_duration_ms,
                stt_finalization_ms=turn.stt_finalization_ms,
                llm_ttft_ms=turn.llm_ttft_ms,
                tts_ttfb_ms=turn.tts_ttfb_ms,
            )
            return turn

    async def attach_assistant_text(
        self,
        assistant_text: str,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_where(lambda c: not c.assistant_text)
            if not turn:
                return None
            turn.assistant_text = assistant_text
            turn.response_text = assistant_text
            return turn

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    async def maybe_finalize(self, trace_turn: Optional[TraceTurn]) -> None:
        if not trace_turn:
            return

        completed_turn: Optional[TraceTurn] = None
        schedule_timeout_for_turn: Optional[str] = None
        async with self._trace_lock:
            if trace_turn not in self._pending_trace_turns:
                return
            if not self._is_complete(trace_turn):
                if self._should_schedule_finalize_timeout(trace_turn):
                    schedule_timeout_for_turn = trace_turn.turn_id
            else:
                completed_turn = self._finalize_locked(trace_turn)

        if schedule_timeout_for_turn:
            self._schedule_finalize_timeout(schedule_timeout_for_turn)
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

    def _is_complete(self, turn: TraceTurn) -> bool:
        return bool(
            turn.user_transcript
            and turn.assistant_text
            and turn.llm_duration_ms is not None
            and turn.tts_duration_ms is not None
        )

    def _should_schedule_finalize_timeout(self, turn: TraceTurn) -> bool:
        return bool(
            turn.llm_duration_ms is not None
            and turn.tts_duration_ms is not None
            and not turn.assistant_text
            and turn.turn_id not in self._trace_finalize_tasks
            and self._trace_finalize_timeout_sec > 0.0
        )

    def _finalize_locked(
        self,
        turn: TraceTurn,
        *,
        missing_assistant_fallback: bool = False,
    ) -> TraceTurn:
        if not turn.prompt_text:
            turn.prompt_text = turn.user_transcript
        if not turn.response_text and turn.assistant_text:
            turn.response_text = turn.assistant_text
        if not turn.assistant_text and turn.response_text:
            turn.assistant_text = turn.response_text

        if missing_assistant_fallback and not turn.assistant_text:
            fallback = self._best_available_assistant_text(turn)
            if fallback:
                turn.assistant_text = fallback
                if not turn.response_text:
                    turn.response_text = fallback
            else:
                turn.assistant_text_missing = True
                unavailable = "[assistant text unavailable]"
                turn.assistant_text = unavailable
                if not turn.response_text:
                    turn.response_text = unavailable

        self._pending_trace_turns.remove(turn)
        self._cancel_finalize_timeout(turn.turn_id)
        return turn

    def _best_available_assistant_text(self, turn: TraceTurn) -> str:
        if turn.assistant_text.strip():
            return turn.assistant_text.strip()
        if turn.response_text.strip():
            return turn.response_text.strip()
        if self._pending_agent_transcripts:
            return self._pending_agent_transcripts.popleft().strip()
        return ""

    # ------------------------------------------------------------------
    # Timeout scheduling
    # ------------------------------------------------------------------

    def _schedule_finalize_timeout(self, turn_id: str) -> None:
        if turn_id in self._trace_finalize_tasks:
            return
        task = asyncio.create_task(self._finalize_after_timeout(turn_id))
        self._trace_finalize_tasks[turn_id] = task
        task.add_done_callback(
            lambda _: self._trace_finalize_tasks.pop(turn_id, None)
        )

    def _cancel_finalize_timeout(self, turn_id: str) -> None:
        task = self._trace_finalize_tasks.pop(turn_id, None)
        current = asyncio.current_task()
        if task and not task.done() and task is not current:
            task.cancel()

    async def _finalize_after_timeout(self, turn_id: str) -> None:
        await asyncio.sleep(self._trace_finalize_timeout_sec)

        completed_turn: Optional[TraceTurn] = None
        async with self._trace_lock:
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
                and pending_turn.llm_duration_ms is not None
                and pending_turn.tts_duration_ms is not None
            ):
                completed_turn = self._finalize_locked(
                    pending_turn, missing_assistant_fallback=True
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

        import src.agent.metrics_collector as _mc

        _tracer = _mc.tracer

        try:
            vals = _prepare_span_values(turn)
            trace_output = turn.assistant_text or turn.response_text

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

                cursor_ns = _emit_component_span(
                    _tracer,
                    name="user_input",
                    context=ctx,
                    start_ns=cursor_ns,
                    duration_ms=vals["user_input_duration_ms"],
                    attributes={"user_transcript": turn.user_transcript},
                    observation_input=turn.user_transcript,
                )
                vad_start_ns = cursor_ns
                cursor_ns = _emit_component_span(
                    _tracer,
                    name="vad",
                    context=ctx,
                    start_ns=cursor_ns,
                    duration_ms=vals["vad_duration_ms"],
                    attributes={"eou_delay_ms": vals["vad_duration_ms"]},
                    observation_output=str(vals["vad_duration_ms"]),
                )
                stt_end_ns = _emit_component_span(
                    _tracer,
                    name="stt",
                    context=ctx,
                    start_ns=vad_start_ns,
                    duration_ms=vals["stt_span_duration_ms"],
                    attributes={
                        "user_transcript": turn.user_transcript,
                        "stt_status": turn.stt_status,
                        "stt_processing_ms": vals["stt_processing_ms"],
                        "stt_finalization_ms": vals["stt_finalization_ms"],
                        "stt_total_latency_ms": vals["stt_total_latency_ms"],
                    },
                    observation_output=turn.user_transcript,
                )
                cursor_ns = max(cursor_ns, stt_end_ns)
                cursor_ns = _emit_component_span(
                    _tracer,
                    name="llm",
                    context=ctx,
                    start_ns=cursor_ns,
                    duration_ms=vals["llm_duration_ms"],
                    attributes={
                        "prompt_text": turn.prompt_text,
                        "response_text": turn.response_text,
                        "ttft_ms": vals["llm_ttft_ms"],
                        "llm_total_latency_ms": vals["llm_total_latency_ms"],
                    },
                    observation_input=turn.prompt_text,
                    observation_output=turn.response_text,
                )
                cursor_ns = _emit_component_span(
                    _tracer,
                    name="tts",
                    context=ctx,
                    start_ns=cursor_ns,
                    duration_ms=vals["tts_duration_ms"],
                    attributes={
                        "assistant_text": turn.assistant_text,
                        "assistant_text_missing": turn.assistant_text_missing,
                        "ttfb_ms": vals["tts_ttfb_ms"],
                    },
                    observation_input=turn.assistant_text,
                    observation_output=turn.assistant_text,
                )
                conv_ms = vals["conversational_latency_ms"]
                if conv_ms is not None:
                    _emit_component_span(
                        _tracer,
                        name="conversation_latency",
                        context=ctx,
                        start_ns=vad_start_ns,
                        duration_ms=conv_ms,
                        attributes={
                            "speech_end_to_assistant_speech_start_ms": conv_ms,
                            "eou_delay_ms": vals["vad_duration_ms"],
                            "stt_finalization_ms": vals["stt_finalization_ms"],
                            "llm_ttft_ms": vals["llm_ttft_ms"],
                            "llm_to_tts_handoff_ms": vals["llm_to_tts_handoff_ms"],
                            "tts_ttfb_ms": vals["tts_ttfb_ms"],
                        },
                        observation_output=str(conv_ms),
                    )
                handoff_ms = vals["llm_to_tts_handoff_ms"]
                if handoff_ms is not None and handoff_ms > 0:
                    handoff_start_ns = vad_start_ns + _ms_to_ns(
                        max(vals["vad_duration_ms"], 0.0)
                        + max(vals["stt_finalization_ms"] or 0.0, 0.0)
                        + max(vals["llm_ttft_ms"], 0.0)
                    )
                    _emit_component_span(
                        _tracer,
                        name="llm_to_tts_handoff",
                        context=ctx,
                        start_ns=handoff_start_ns,
                        duration_ms=handoff_ms,
                        attributes={
                            "llm_to_tts_handoff_ms": handoff_ms,
                            "speech_end_to_assistant_speech_start_ms": conv_ms,
                            "eou_delay_ms": vals["vad_duration_ms"],
                            "stt_finalization_ms": vals["stt_finalization_ms"],
                            "llm_ttft_ms": vals["llm_ttft_ms"],
                            "tts_ttfb_ms": vals["tts_ttfb_ms"],
                        },
                        observation_output=str(handoff_ms),
                    )
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


def _recompute_conversational_latency(turn: TraceTurn) -> None:
    turn.conversational_latency_ms = _compute_conversational_latency_ms(
        vad_duration_ms=turn.vad_duration_ms,
        stt_finalization_ms=turn.stt_finalization_ms,
        llm_ttft_ms=turn.llm_ttft_ms,
        tts_ttfb_ms=turn.tts_ttfb_ms,
    )


def _compute_conversational_latency_ms(
    *,
    vad_duration_ms: Optional[float],
    stt_finalization_ms: Optional[float],
    llm_ttft_ms: Optional[float],
    tts_ttfb_ms: Optional[float],
) -> Optional[float]:
    components = (vad_duration_ms, stt_finalization_ms, llm_ttft_ms, tts_ttfb_ms)
    if any(c is None for c in components):
        return None
    return sum(c for c in components if c is not None)


def _compute_llm_to_tts_handoff_ms(
    *,
    total_latency_ms: Optional[float],
    vad_duration_ms: Optional[float],
    stt_finalization_ms: Optional[float],
    llm_ttft_ms: Optional[float],
    tts_ttfb_ms: Optional[float],
) -> Optional[float]:
    if total_latency_ms is None:
        return None
    baseline = _compute_conversational_latency_ms(
        vad_duration_ms=vad_duration_ms,
        stt_finalization_ms=stt_finalization_ms,
        llm_ttft_ms=llm_ttft_ms,
        tts_ttfb_ms=tts_ttfb_ms,
    )
    if baseline is None:
        return None
    return max(total_latency_ms - baseline, 0.0)


def _total_duration_ms(turn: TraceTurn) -> float:
    stt = (
        turn.stt_finalization_ms
        if turn.stt_finalization_ms is not None
        else (turn.stt_duration_ms if turn.stt_duration_ms is not None else 0.0)
    )
    llm = (
        turn.llm_total_latency_ms
        if turn.llm_total_latency_ms is not None
        else (turn.llm_duration_ms or 0.0)
    )
    calculated = (
        (turn.vad_duration_ms or 0.0) + stt + llm + (turn.tts_duration_ms or 0.0)
    )
    if turn.conversational_latency_ms is not None:
        calculated = max(calculated, turn.conversational_latency_ms)
    return calculated


def _prepare_span_values(turn: TraceTurn) -> dict[str, Any]:
    """Pre-compute derived values used by span emission."""
    user_input_duration_ms = 0.0 if turn.user_transcript else None
    vad_duration_ms = max(turn.vad_duration_ms or 0.0, 0.0)
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
    if stt_total_latency_ms is not None and stt_total_latency_ms > 0:
        stt_span_duration_ms = stt_total_latency_ms
    elif stt_finalization_ms is not None and stt_finalization_ms > 0:
        stt_span_duration_ms = stt_finalization_ms
    else:
        stt_span_duration_ms = stt_processing_ms

    llm_duration_ms = max(turn.llm_duration_ms or 0.0, 0.0)
    llm_ttft_ms = max(turn.llm_ttft_ms or 0.0, 0.0)
    llm_total_latency_ms = (
        max(turn.llm_total_latency_ms, 0.0)
        if turn.llm_total_latency_ms is not None
        else llm_duration_ms
    )
    tts_duration_ms = max(turn.tts_duration_ms or 0.0, 0.0)
    tts_ttfb_ms = max(turn.tts_ttfb_ms or 0.0, 0.0)
    conversational_latency_ms = (
        max(turn.conversational_latency_ms, 0.0)
        if turn.conversational_latency_ms is not None
        else None
    )
    llm_to_tts_handoff_ms = (
        max(turn.llm_to_tts_handoff_ms, 0.0)
        if turn.llm_to_tts_handoff_ms is not None
        else None
    )
    return {
        "user_input_duration_ms": user_input_duration_ms,
        "vad_duration_ms": vad_duration_ms,
        "stt_processing_ms": stt_processing_ms,
        "stt_finalization_ms": stt_finalization_ms,
        "stt_total_latency_ms": stt_total_latency_ms,
        "stt_span_duration_ms": stt_span_duration_ms,
        "llm_duration_ms": llm_duration_ms,
        "llm_ttft_ms": llm_ttft_ms,
        "llm_total_latency_ms": llm_total_latency_ms,
        "tts_duration_ms": tts_duration_ms,
        "tts_ttfb_ms": tts_ttfb_ms,
        "conversational_latency_ms": conversational_latency_ms,
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
        "duration_ms": _total_duration_ms(turn),
        "latency_ms.user_input": vals["user_input_duration_ms"],
        "latency_ms.vad": vals["vad_duration_ms"],
        "latency_ms.eou_delay": vals["vad_duration_ms"],
        "latency_ms.stt": vals["stt_span_duration_ms"],
        "latency_ms.stt_processing": vals["stt_processing_ms"],
        "latency_ms.stt_finalization": vals["stt_finalization_ms"],
        "latency_ms.stt_total": vals["stt_total_latency_ms"],
        "latency_ms.llm": vals["llm_duration_ms"],
        "latency_ms.llm_ttft": vals["llm_ttft_ms"],
        "latency_ms.llm_total": vals["llm_total_latency_ms"],
        "latency_ms.tts": vals["tts_duration_ms"],
        "latency_ms.tts_ttfb": vals["tts_ttfb_ms"],
        "latency_ms.llm_to_tts_handoff": vals["llm_to_tts_handoff_ms"],
        "latency_ms.conversational": vals["conversational_latency_ms"],
        "latency_ms.speech_end_to_assistant_speech_start": vals[
            "conversational_latency_ms"
        ],
        "stt_status": turn.stt_status,
    }
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


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
