"""Metrics collector for LiveKit agent telemetry.

Aggregates metrics from AgentSession events and publishes to data channel for
real-time monitoring. Also creates one Langfuse trace per finalized user turn.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from time import time, time_ns
from typing import Any, Optional, Union

from livekit import rtc
from livekit.agents import metrics
from livekit.agents.telemetry import tracer
from opentelemetry import trace

from src.core.logger import logger


@dataclass
class STTMetrics:
    """Speech-to-text metrics."""

    model_name: str
    audio_duration: float
    duration: float


@dataclass
class LLMMetrics:
    """Language model metrics."""

    ttft: float  # Time to first token
    duration: float
    tokens: int
    tokens_per_second: float


@dataclass
class TTSMetrics:
    """Text-to-speech metrics."""

    ttfb: float  # Time to first byte
    duration: float
    audio_duration: float


@dataclass
class VADMetrics:
    """Voice activity detection metrics."""

    idle_time: float
    inference_duration_total: float
    inference_count: int


@dataclass
class Latencies:
    """Latency breakdown for conversation turn."""

    total_latency: float
    eou_delay: float
    vad_detection_delay: float
    llm_ttft: float
    tts_ttfb: float


@dataclass
class TurnMetrics:
    """Aggregated metrics for a conversation turn."""

    turn_id: str
    timestamp: float
    role: str  # "user" | "agent"
    transcript: str = ""
    stt: Optional[STTMetrics] = None
    llm: Optional[LLMMetrics] = None
    tts: Optional[TTSMetrics] = None
    vad: Optional[VADMetrics] = None
    latencies: Optional[Latencies] = None

    def compute_latencies(self, eou_delay: float = 0.0) -> None:
        """Compute total latency from component metrics.

        Total latency = eou_delay + llm_ttft + tts_ttfb
        """
        llm_ttft = self.llm.ttft if self.llm else 0.0
        tts_ttfb = self.tts.ttfb if self.tts else 0.0
        total_latency = eou_delay + llm_ttft + tts_ttfb

        self.latencies = Latencies(
            total_latency=total_latency,
            eou_delay=eou_delay,
            vad_detection_delay=eou_delay,
            llm_ttft=llm_ttft,
            tts_ttfb=tts_ttfb,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
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
                "llm": asdict(self.llm) if self.llm else None,
                "tts": asdict(self.tts) if self.tts else None,
                "vad": asdict(self.vad) if self.vad else None,
            },
            "latencies": asdict(self.latencies) if self.latencies else None,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


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
    vad_duration_ms: Optional[float] = None
    stt_duration_ms: Optional[float] = None
    llm_duration_ms: Optional[float] = None
    tts_duration_ms: Optional[float] = None
    trace_id: Optional[str] = None


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
        langfuse_enabled: bool = False,
    ) -> None:
        """Initialize metrics collector.

        Args:
            room: LiveKit room for publishing metrics
            model_name: STT model name to include in metrics
            room_name: LiveKit room name
            room_id: LiveKit room id (sid) when available
            participant_id: LiveKit participant identity when available
            langfuse_enabled: Enable one-trace-per-turn Langfuse traces
        """
        self._room = room
        self._model_name = model_name
        self._pending_metrics: dict[str, TurnMetrics] = {}
        self._eou_delays: dict[str, float] = {}
        self._pending_transcripts: deque[str] = deque()
        self._pending_agent_transcripts: deque[str] = deque()
        self._latest_agent_speech_id: Optional[str] = None

        self._room_name = room_name or self.UNKNOWN_ROOM_ID
        self._room_id = room_id or room_name or self.UNKNOWN_ROOM_ID
        self._session_id = self.UNKNOWN_SESSION_ID
        self._participant_id = participant_id or self.UNKNOWN_PARTICIPANT_ID
        self._langfuse_enabled = langfuse_enabled
        self._pending_trace_turns: deque[TraceTurn] = deque()
        self._trace_lock = asyncio.Lock()

    async def on_session_metadata(
        self,
        *,
        session_id: Any,
        participant_id: Any,
    ) -> None:
        """Update session context received from Streamlit/LiveKit data channel."""
        normalized_session_id = self._normalize_optional_text(session_id)
        normalized_participant_id = self._normalize_optional_text(participant_id)

        async with self._trace_lock:
            if normalized_session_id:
                self._session_id = normalized_session_id
            if normalized_participant_id:
                self._participant_id = normalized_participant_id

            for turn in self._pending_trace_turns:
                if (
                    turn.session_id == self.UNKNOWN_SESSION_ID
                    and self._session_id != self.UNKNOWN_SESSION_ID
                ):
                    turn.session_id = self._session_id
                if (
                    turn.participant_id == self.UNKNOWN_PARTICIPANT_ID
                    and self._participant_id != self.UNKNOWN_PARTICIPANT_ID
                ):
                    turn.participant_id = self._participant_id

    async def on_user_input_transcribed(
        self,
        transcript: str,
        *,
        is_final: bool,
    ) -> None:
        """Store final user transcripts and create a new trace turn."""
        if not is_final:
            return
        normalized = transcript.strip()
        if not normalized:
            return
        self._pending_transcripts.append(normalized)
        room_id = await self._resolve_room_id()

        async with self._trace_lock:
            self._pending_trace_turns.append(
                TraceTurn(
                    turn_id=str(uuid.uuid4()),
                    session_id=self._session_id,
                    room_id=room_id,
                    participant_id=self._participant_id,
                    user_transcript=normalized,
                    prompt_text=normalized,
                    # Some STT providers emit final transcript events without STT metrics.
                    # Keep one-trace-per-turn behavior by defaulting STT duration to 0ms.
                    stt_duration_ms=0.0,
                )
            )

    async def on_conversation_item_added(
        self,
        *,
        role: Optional[str],
        content: Any,
    ) -> None:
        """Store conversation text from chat items for history rendering."""
        if role not in {"user", "assistant"}:
            return
        normalized = self._extract_content_text(content).strip()
        if not normalized:
            return
        if role == "user":
            self._append_if_new(self._pending_transcripts, normalized)
            return

        self._append_if_new(self._pending_agent_transcripts, normalized)
        trace_turn = await self._attach_assistant_text(normalized)
        await self._maybe_finalize_trace_turn(trace_turn)

    async def on_tts_synthesized(
        self,
        *,
        ttfb: float,
        duration: float,
        audio_duration: float,
    ) -> None:
        """Handle fallback TTS metrics from custom PocketTTS stream."""
        if ttfb < 0:
            return
        speech_id = self._latest_agent_speech_id or f"tts-{uuid.uuid4()}"
        turn_metrics = self._get_or_create_turn(speech_id, role="agent")
        turn_metrics.tts = TTSMetrics(
            ttfb=ttfb,
            duration=duration,
            audio_duration=audio_duration,
        )
        await self._publish_live_update(
            speech_id=speech_id,
            stage="tts",
            role=turn_metrics.role,
            turn_metrics=turn_metrics,
        )
        logger.debug(
            "TTS fallback metrics collected: "
            f"speech_id={speech_id}, ttfb={ttfb:.3f}s"
        )
        await self._maybe_publish_turn(speech_id, turn_metrics)

        trace_turn = await self._attach_tts_stage(duration=duration, fallback_duration=audio_duration)
        await self._maybe_finalize_trace_turn(trace_turn)

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
        """Handle metrics collected event from AgentSession.

        Aggregates metrics by speech_id and publishes when turn is complete.

        Args:
            collected_metrics: Metrics event from AgentSession
        """
        speech_id = None
        turn_metrics = None
        trace_turn = None

        # Extract metrics based on type
        if isinstance(collected_metrics, metrics.STTMetrics):
            # STTMetrics doesn't have speech_id, use request_id as correlation key
            speech_id = collected_metrics.request_id
            turn_metrics = self._get_or_create_turn(speech_id, role="user")
            if self._pending_transcripts:
                turn_metrics.transcript = self._pending_transcripts.popleft()
            # STTMetrics doesn't have transcript field - it comes from the STT events, not metrics
            turn_metrics.stt = STTMetrics(
                model_name=self._model_name,
                audio_duration=collected_metrics.audio_duration,
                duration=collected_metrics.duration,
            )
            await self._publish_live_update(
                speech_id=speech_id,
                stage="stt",
                role=turn_metrics.role,
                turn_metrics=turn_metrics,
            )
            logger.debug(
                f"STT metrics collected: request_id={speech_id}, duration={collected_metrics.duration:.3f}s"
            )
            trace_turn = await self._attach_stt_stage(
                transcript=turn_metrics.transcript,
                duration=collected_metrics.duration,
                fallback_duration=collected_metrics.audio_duration,
            )

        elif isinstance(collected_metrics, metrics.LLMMetrics):
            speech_id = collected_metrics.speech_id or collected_metrics.request_id
            turn_metrics = self._get_or_create_turn(speech_id, role="agent")
            self._latest_agent_speech_id = speech_id
            if self._pending_agent_transcripts and not turn_metrics.transcript:
                turn_metrics.transcript = self._pending_agent_transcripts.popleft()
            turn_metrics.llm = LLMMetrics(
                ttft=collected_metrics.ttft,
                duration=collected_metrics.duration,
                tokens=collected_metrics.completion_tokens,
                tokens_per_second=(
                    collected_metrics.completion_tokens / collected_metrics.duration
                    if collected_metrics.duration > 0
                    else 0.0
                ),
            )
            await self._publish_live_update(
                speech_id=speech_id,
                stage="llm",
                role=turn_metrics.role,
                turn_metrics=turn_metrics,
            )
            logger.debug(
                f"LLM metrics collected: speech_id={speech_id}, ttft={collected_metrics.ttft:.3f}s"
            )
            trace_turn = await self._attach_llm_stage(duration=collected_metrics.duration)

        elif isinstance(collected_metrics, metrics.TTSMetrics):
            speech_id = collected_metrics.speech_id or collected_metrics.request_id
            turn_metrics = self._get_or_create_turn(speech_id, role="agent")
            turn_metrics.tts = TTSMetrics(
                ttfb=collected_metrics.ttfb,
                duration=collected_metrics.duration,
                audio_duration=collected_metrics.audio_duration,
            )
            await self._publish_live_update(
                speech_id=speech_id,
                stage="tts",
                role=turn_metrics.role,
                turn_metrics=turn_metrics,
            )
            logger.debug(
                f"TTS metrics collected: speech_id={speech_id}, ttfb={collected_metrics.ttfb:.3f}s"
            )
            trace_turn = await self._attach_tts_stage(
                duration=collected_metrics.duration,
                fallback_duration=collected_metrics.audio_duration,
            )

        elif isinstance(collected_metrics, metrics.EOUMetrics):
            speech_id = collected_metrics.speech_id
            if speech_id:
                self._eou_delays[speech_id] = collected_metrics.end_of_utterance_delay
                turn_metrics = self._pending_metrics.get(speech_id)
                await self._publish_live_update(
                    speech_id=speech_id,
                    stage="eou",
                    role=turn_metrics.role if turn_metrics else None,
                    turn_metrics=turn_metrics,
                )
                logger.debug(
                    f"EOU metrics collected: speech_id={speech_id}, delay={collected_metrics.end_of_utterance_delay:.3f}s"
                )
                trace_turn = await self._attach_vad_stage(
                    duration=collected_metrics.end_of_utterance_delay
                )

        elif isinstance(collected_metrics, metrics.VADMetrics):
            # Some SDK versions don't include speech_id on VAD metrics
            speech_id = getattr(collected_metrics, "speech_id", None)
            turn_metrics = self._pending_metrics.get(speech_id) if speech_id else None
            if speech_id and turn_metrics:
                turn_metrics.vad = VADMetrics(
                    idle_time=collected_metrics.idle_time,
                    inference_duration_total=collected_metrics.inference_duration_total,
                    inference_count=collected_metrics.inference_count,
                )
            await self._publish_live_update(
                speech_id=speech_id,
                stage="vad",
                role=turn_metrics.role if turn_metrics else None,
                turn_metrics=turn_metrics,
                vad_metrics=VADMetrics(
                    idle_time=collected_metrics.idle_time,
                    inference_duration_total=collected_metrics.inference_duration_total,
                    inference_count=collected_metrics.inference_count,
                ),
                diagnostic=not bool(speech_id and turn_metrics),
            )
            logger.debug(
                "VAD metrics collected: "
                f"speech_id={speech_id or 'n/a'}, idle_time={collected_metrics.idle_time:.3f}s"
            )

        # Check if turn is complete and publish
        if speech_id and turn_metrics:
            await self._maybe_publish_turn(speech_id, turn_metrics)
        await self._maybe_finalize_trace_turn(trace_turn)

    def _get_or_create_turn(self, speech_id: str, role: str) -> TurnMetrics:
        """Get existing turn metrics or create new one.

        Args:
            speech_id: Unique identifier for conversation turn
            role: Role for this turn ("user" or "agent")

        Returns:
            TurnMetrics instance
        """
        if speech_id not in self._pending_metrics:
            self._pending_metrics[speech_id] = TurnMetrics(
                turn_id=str(uuid.uuid4()),
                timestamp=time(),
                role=role,
            )
        return self._pending_metrics[speech_id]

    async def _maybe_publish_turn(
        self, speech_id: str, turn_metrics: TurnMetrics
    ) -> None:
        """Publish turn metrics if all required metrics are available.

        For user turns: requires STT
        For agent turns: requires LLM and TTS

        Args:
            speech_id: Unique identifier for conversation turn
            turn_metrics: Aggregated turn metrics
        """
        is_complete = False

        if turn_metrics.role == "user" and turn_metrics.stt:
            is_complete = True
        elif (
            turn_metrics.role == "agent"
            and turn_metrics.llm
            and turn_metrics.tts
        ):
            is_complete = True

        if is_complete:
            # Compute latencies
            eou_delay = self._eou_delays.get(speech_id, 0.0)
            turn_metrics.compute_latencies(eou_delay)

            # Publish to data channel
            await self._publish_metrics(turn_metrics)

            # Cleanup
            self._pending_metrics.pop(speech_id, None)
            self._eou_delays.pop(speech_id, None)

    async def _publish_live_update(
        self,
        *,
        speech_id: Optional[str],
        stage: str,
        role: Optional[str],
        turn_metrics: Optional[TurnMetrics],
        vad_metrics: Optional[VADMetrics] = None,
        diagnostic: bool = False,
    ) -> None:
        """Publish ephemeral live update for immediate frontend rendering."""
        try:
            payload: dict[str, Any] = {
                "type": "metrics_live_update",
                "timestamp": time(),
                "speech_id": speech_id,
                "stage": stage,
                "role": role,
                "metrics": {
                    "stt": None,
                    "llm": None,
                    "tts": None,
                    "vad": None,
                },
                "latencies": None,
                "diagnostic": diagnostic,
            }

            if turn_metrics:
                if turn_metrics.stt:
                    payload["metrics"]["stt"] = {
                        **asdict(turn_metrics.stt),
                        "display_duration": self._stt_display_duration(turn_metrics.stt),
                    }
                if turn_metrics.llm:
                    payload["metrics"]["llm"] = asdict(turn_metrics.llm)
                if turn_metrics.tts:
                    payload["metrics"]["tts"] = asdict(turn_metrics.tts)
                if turn_metrics.vad:
                    payload["metrics"]["vad"] = asdict(turn_metrics.vad)
                payload["latencies"] = self._build_partial_latencies(
                    speech_id=speech_id,
                    turn_metrics=turn_metrics,
                )

            if vad_metrics:
                payload["metrics"]["vad"] = asdict(vad_metrics)

            await self._room.local_participant.publish_data(
                payload=json.dumps(payload).encode("utf-8"),
                topic="metrics",
                reliable=True,
            )
        except Exception as e:
            logger.error(f"Failed to publish live metrics update: {e}")

    def _build_partial_latencies(
        self,
        *,
        speech_id: Optional[str],
        turn_metrics: TurnMetrics,
    ) -> Optional[dict[str, float]]:
        """Build a latency preview for live updates."""
        has_component = bool(turn_metrics.llm or turn_metrics.tts)
        if not has_component:
            return None

        eou_delay = self._eou_delays.get(speech_id or "", 0.0)
        llm_ttft = turn_metrics.llm.ttft if turn_metrics.llm else 0.0
        tts_ttfb = turn_metrics.tts.ttfb if turn_metrics.tts else 0.0
        return {
            "total_latency": eou_delay + llm_ttft + tts_ttfb,
            "eou_delay": eou_delay,
            "vad_detection_delay": eou_delay,
            "llm_ttft": llm_ttft,
            "tts_ttfb": tts_ttfb,
        }

    def _stt_display_duration(self, stt_metrics: STTMetrics) -> float:
        """Prefer measured STT duration, fallback to audio duration if missing."""
        if stt_metrics.duration > 0:
            return stt_metrics.duration
        return stt_metrics.audio_duration

    async def _publish_metrics(self, turn_metrics: TurnMetrics) -> None:
        """Publish metrics to LiveKit data channel.

        Args:
            turn_metrics: Turn metrics to publish
        """
        try:
            json_data = turn_metrics.to_json()
            await self._room.local_participant.publish_data(
                payload=json_data.encode("utf-8"),
                topic="metrics",
                reliable=True,
            )
            logger.info(
                f"Published metrics for turn_id={turn_metrics.turn_id}, role={turn_metrics.role}, "
                f"total_latency={turn_metrics.latencies.total_latency:.3f}s"
            )
        except Exception as e:
            logger.error(f"Failed to publish metrics: {e}")

    async def _attach_stt_stage(
        self,
        *,
        transcript: str,
        duration: float,
        fallback_duration: float,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_for_stt()
            if not turn:
                return None
            if transcript:
                turn.user_transcript = transcript.strip()
            turn.prompt_text = turn.user_transcript
            turn.stt_duration_ms = self._duration_to_ms(duration, fallback_duration)
            return turn

    async def _attach_vad_stage(self, *, duration: float) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_for_vad()
            if not turn:
                return None
            turn.vad_duration_ms = self._duration_to_ms(duration, 0.0)
            return turn

    async def _attach_llm_stage(self, *, duration: float) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_for_llm()
            if not turn:
                return None
            turn.prompt_text = turn.user_transcript
            turn.llm_duration_ms = self._duration_to_ms(duration, 0.0)
            return turn

    async def _attach_tts_stage(
        self,
        *,
        duration: float,
        fallback_duration: float,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_for_tts()
            if not turn:
                return None
            turn.tts_duration_ms = self._duration_to_ms(duration, fallback_duration)
            return turn

    async def _attach_assistant_text(self, assistant_text: str) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_for_assistant_text()
            if not turn:
                return None
            turn.assistant_text = assistant_text
            turn.response_text = assistant_text
            return turn

    async def _maybe_finalize_trace_turn(self, trace_turn: Optional[TraceTurn]) -> None:
        if not trace_turn:
            return

        completed_turn: Optional[TraceTurn] = None
        async with self._trace_lock:
            if trace_turn not in self._pending_trace_turns:
                return
            if not self._is_trace_turn_complete(trace_turn):
                return

            if not trace_turn.prompt_text:
                trace_turn.prompt_text = trace_turn.user_transcript
            if not trace_turn.response_text and trace_turn.assistant_text:
                trace_turn.response_text = trace_turn.assistant_text
            if not trace_turn.assistant_text and trace_turn.response_text:
                trace_turn.assistant_text = trace_turn.response_text

            self._pending_trace_turns.remove(trace_turn)
            completed_turn = trace_turn

        if completed_turn:
            await self._emit_turn_trace(completed_turn)
            await self._publish_trace_update(completed_turn)

    def _next_turn_for_stt(self) -> Optional[TraceTurn]:
        for turn in self._pending_trace_turns:
            if turn.stt_duration_ms is None or turn.stt_duration_ms == 0.0:
                return turn
        return None

    def _next_turn_for_vad(self) -> Optional[TraceTurn]:
        for turn in self._pending_trace_turns:
            if turn.stt_duration_ms is not None and turn.vad_duration_ms is None:
                return turn
        return None

    def _next_turn_for_llm(self) -> Optional[TraceTurn]:
        for turn in self._pending_trace_turns:
            if turn.stt_duration_ms is not None and turn.llm_duration_ms is None:
                return turn
        return None

    def _next_turn_for_tts(self) -> Optional[TraceTurn]:
        for turn in self._pending_trace_turns:
            if turn.llm_duration_ms is not None and turn.tts_duration_ms is None:
                return turn
        return None

    def _next_turn_for_assistant_text(self) -> Optional[TraceTurn]:
        for turn in self._pending_trace_turns:
            if not turn.assistant_text:
                return turn
        return None

    def _is_trace_turn_complete(self, turn: TraceTurn) -> bool:
        return bool(
            turn.user_transcript
            and turn.stt_duration_ms is not None
            and turn.llm_duration_ms is not None
            and turn.tts_duration_ms is not None
        )

    async def _emit_turn_trace(self, turn: TraceTurn) -> None:
        if not self._langfuse_enabled:
            return

        try:
            user_input_duration_ms = 1.0 if turn.user_transcript else 0.0
            vad_duration_ms = max(turn.vad_duration_ms or 0.0, 0.0)
            stt_duration_ms = max(turn.stt_duration_ms or 0.0, 0.0)
            llm_duration_ms = max(turn.llm_duration_ms or 0.0, 0.0)
            tts_duration_ms = max(turn.tts_duration_ms or 0.0, 0.0)

            root_context = trace.set_span_in_context(trace.INVALID_SPAN)
            root_start_ns = time_ns()
            cursor_ns = root_start_ns

            with tracer.start_as_current_span(
                "turn",
                context=root_context,
                start_time=root_start_ns,
            ) as turn_span:
                turn.trace_id = trace.format_trace_id(turn_span.get_span_context().trace_id)
                turn_span.set_attribute("session_id", turn.session_id)
                turn_span.set_attribute("room_id", turn.room_id)
                turn_span.set_attribute("participant_id", turn.participant_id)
                turn_span.set_attribute("turn_id", turn.turn_id)
                turn_span.set_attribute("langfuse.session.id", turn.session_id)
                turn_span.set_attribute("session.id", turn.session_id)
                turn_span.set_attribute("langfuse.user.id", turn.participant_id)
                turn_span.set_attribute("user.id", turn.participant_id)
                turn_span.set_attribute("langfuse.trace.name", "turn")
                turn_span.set_attribute("langfuse.trace.input", turn.user_transcript)
                turn_span.set_attribute(
                    "langfuse.trace.output",
                    turn.assistant_text or turn.response_text,
                )
                turn_span.set_attribute("langfuse.trace.metadata.room_id", turn.room_id)
                turn_span.set_attribute(
                    "langfuse.trace.metadata.participant_id",
                    turn.participant_id,
                )
                turn_span.set_attribute("langfuse.trace.metadata.turn_id", turn.turn_id)
                turn_span.set_attribute("duration_ms", self._total_duration_ms(turn))
                turn_span.set_attribute("latency_ms.user_input", user_input_duration_ms)
                turn_span.set_attribute("latency_ms.vad", vad_duration_ms)
                turn_span.set_attribute("latency_ms.stt", stt_duration_ms)
                turn_span.set_attribute("latency_ms.llm", llm_duration_ms)
                turn_span.set_attribute("latency_ms.tts", tts_duration_ms)

                component_context = trace.set_span_in_context(turn_span)

                cursor_ns = self._emit_component_span(
                    name="user_input",
                    context=component_context,
                    start_ns=cursor_ns,
                    duration_ms=user_input_duration_ms,
                    attributes={"user_transcript": turn.user_transcript},
                )
                cursor_ns = self._emit_component_span(
                    name="vad",
                    context=component_context,
                    start_ns=cursor_ns,
                    duration_ms=vad_duration_ms,
                    attributes={"eou_delay_ms": vad_duration_ms},
                )
                cursor_ns = self._emit_component_span(
                    name="stt",
                    context=component_context,
                    start_ns=cursor_ns,
                    duration_ms=stt_duration_ms,
                    attributes={"user_transcript": turn.user_transcript},
                )
                cursor_ns = self._emit_component_span(
                    name="llm",
                    context=component_context,
                    start_ns=cursor_ns,
                    duration_ms=llm_duration_ms,
                    attributes={
                        "prompt_text": turn.prompt_text,
                        "response_text": turn.response_text,
                    },
                )
                cursor_ns = self._emit_component_span(
                    name="tts",
                    context=component_context,
                    start_ns=cursor_ns,
                    duration_ms=tts_duration_ms,
                    attributes={"assistant_text": turn.assistant_text},
                )

                self._close_span_at(turn_span, cursor_ns)
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

    def _emit_component_span(
        self,
        *,
        name: str,
        context: Any,
        start_ns: int,
        duration_ms: float,
        attributes: dict[str, Any],
    ) -> int:
        actual_duration_ms = max(duration_ms, 0.0)
        timeline_duration_ms = self._timeline_duration_ms(actual_duration_ms)
        end_ns = start_ns + self._duration_ms_to_ns(timeline_duration_ms)

        with tracer.start_as_current_span(name, context=context, start_time=start_ns) as span:
            span.set_attribute("duration_ms", actual_duration_ms)
            for key, value in attributes.items():
                if value is None:
                    continue
                span.set_attribute(key, value)
            self._close_span_at(span, end_ns)
        return end_ns

    def _close_span_at(self, span: Any, end_ns: int) -> None:
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

    def _timeline_duration_ms(self, duration_ms: float) -> float:
        """Ensure zero-duration stages are still visible in timeline views."""
        return duration_ms if duration_ms > 0 else 1.0

    def _duration_ms_to_ns(self, duration_ms: float) -> int:
        return int(max(duration_ms, 0.0) * 1_000_000)

    async def _flush_tracer_provider(self) -> None:
        """Best-effort flush so traces are visible in Langfuse faster."""
        try:
            tracer_provider = trace.get_tracer_provider()
            force_flush = getattr(tracer_provider, "force_flush", None)
            if not callable(force_flush):
                return
            await asyncio.to_thread(force_flush)
        except Exception as exc:
            logger.debug(f"Failed to force flush tracer provider: {exc}")

    async def _publish_trace_update(self, turn: TraceTurn) -> None:
        if not turn.trace_id:
            return
        try:
            payload = {
                "type": "trace_update",
                "session_id": turn.session_id,
                "turn_id": turn.turn_id,
                "trace_id": turn.trace_id,
                "timestamp": time(),
            }
            await self._room.local_participant.publish_data(
                payload=json.dumps(payload).encode("utf-8"),
                topic="metrics",
                reliable=True,
            )
        except Exception as exc:
            logger.error(f"Failed to publish trace update: {exc}")

    def _total_duration_ms(self, turn: TraceTurn) -> float:
        return (
            (turn.vad_duration_ms or 0.0)
            + (turn.stt_duration_ms or 0.0)
            + (turn.llm_duration_ms or 0.0)
            + (turn.tts_duration_ms or 0.0)
        )

    def _duration_to_ms(self, duration: float, fallback_duration: float) -> float:
        chosen = duration if duration > 0 else fallback_duration
        return max(chosen, 0.0) * 1000.0

    def _normalize_optional_text(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    async def _resolve_room_id(self) -> str:
        if self._room_id and self._room_id != self._room_name:
            return self._room_id
        try:
            sid = await self._room.sid
            normalized = self._normalize_optional_text(sid)
            if normalized:
                self._room_id = normalized
        except Exception:
            pass
        return self._room_id or self._room_name or self.UNKNOWN_ROOM_ID

    def _extract_content_text(self, content: Any) -> str:
        """Extract plain text from a chat content payload."""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return " ".join(part for part in parts if part)

    def _append_if_new(self, queue: deque[str], value: str) -> None:
        """Avoid duplicate adjacent transcript entries."""
        if queue and queue[-1] == value:
            return
        queue.append(value)
