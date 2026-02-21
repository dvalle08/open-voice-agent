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
from time import monotonic, time, time_ns
from typing import Any, Callable, Optional, Sequence, Union

from livekit import rtc
from livekit.agents import metrics
from livekit.agents.telemetry import tracer
from opentelemetry import trace

from src.core.logger import logger
from src.core.settings import settings


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
    role: str  # "user" | "agent"
    transcript: str = ""
    stt: Optional[STTMetrics] = None
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
        """Compute total latency from component metrics.

        Total latency prefers observed wall-clock elapsed time (EOU -> first
        assistant audio) and falls back to component approximation:
        eou_delay + stt_finalization_delay + llm_ttft + tts_ttfb
        """
        llm_ttft = self.llm.ttft if self.llm else 0.0
        tts_ttfb = self.tts.ttfb if self.tts else 0.0
        baseline_latency = eou_delay + stt_finalization_delay + llm_ttft + tts_ttfb
        observed = observed_total_latency if observed_total_latency is not None else 0.0
        total_latency = max(baseline_latency, observed)
        llm_to_tts_handoff_latency = max(total_latency - baseline_latency, 0.0)

        self.latencies = Latencies(
            total_latency=total_latency,
            eou_delay=eou_delay,
            stt_finalization_delay=stt_finalization_delay,
            llm_to_tts_handoff_latency=llm_to_tts_handoff_latency,
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
    assistant_text_missing: bool = False
    stt_status: str = "missing"  # "measured" | "missing"
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


class MetricsCollector:
    """Collects and publishes agent metrics to LiveKit data channel."""

    UNKNOWN_SESSION_ID = "unknown-session"
    UNKNOWN_PARTICIPANT_ID = "unknown-participant"
    UNKNOWN_ROOM_ID = "unknown-room"
    DEFAULT_TRACE_FINALIZE_TIMEOUT_MS = 8000.0
    DEFAULT_MAX_PENDING_TRACE_TASKS = 200
    DEFAULT_TRACE_FLUSH_TIMEOUT_SEC = 1.0

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
        """Initialize metrics collector.

        Args:
            room: LiveKit room for publishing metrics
            model_name: STT model name to include in metrics
            room_name: LiveKit room name
            room_id: LiveKit room id (sid) when available
            participant_id: LiveKit participant identity when available
            fallback_session_prefix: Prefix used for generated fallback session id
                (e.g. "console" -> "console_<uuid>") when no metadata session id exists
            fallback_participant_prefix: Prefix used for generated fallback participant id
                (e.g. "console" -> "console_<uuid>") when no participant identity exists
            langfuse_enabled: Enable one-trace-per-turn Langfuse traces
        """
        self._room = room
        self._model_name = model_name
        self._pending_metrics: dict[str, TurnMetrics] = {}
        self._eou_delays: dict[str, float] = {}
        self._stt_finalization_delays: dict[str, float] = {}
        self._speech_end_monotonic_by_speech: dict[str, float] = {}
        self._first_audio_monotonic_by_speech: dict[str, float] = {}
        self._pending_speech_ids_for_first_audio: deque[str] = deque()
        self._pending_transcripts: deque[str] = deque()
        self._pending_agent_transcripts: deque[str] = deque()
        self._latest_agent_speech_id: Optional[str] = None

        self._room_name = room_name or self.UNKNOWN_ROOM_ID
        self._room_id = room_id or room_name or self.UNKNOWN_ROOM_ID
        self._fallback_session_id = self._build_fallback_id(fallback_session_prefix)
        self._session_id = self._fallback_session_id or self.UNKNOWN_SESSION_ID
        self._fallback_participant_id = self._build_fallback_id(
            fallback_participant_prefix
        )
        self._participant_id = (
            self._normalize_optional_text(participant_id)
            or self._fallback_participant_id
            or self.UNKNOWN_PARTICIPANT_ID
        )
        self._langfuse_enabled = langfuse_enabled
        self._pending_trace_turns: deque[TraceTurn] = deque()
        self._trace_lock = asyncio.Lock()
        self._trace_finalize_timeout_sec = (
            max(
                getattr(
                    settings.langfuse,
                    "LANGFUSE_TRACE_FINALIZE_TIMEOUT_MS",
                    self.DEFAULT_TRACE_FINALIZE_TIMEOUT_MS,
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
                    self.DEFAULT_MAX_PENDING_TRACE_TASKS,
                )
            ),
            1,
        )
        self._trace_flush_timeout_sec = (
            max(
                getattr(
                    settings.langfuse,
                    "LANGFUSE_TRACE_FLUSH_TIMEOUT_MS",
                    self.DEFAULT_TRACE_FLUSH_TIMEOUT_SEC * 1000.0,
                ),
                0.0,
            )
            / 1000.0
        )
        self._trace_emit_tasks: set[asyncio.Task[None]] = set()
        self._trace_finalize_tasks: dict[str, asyncio.Task[None]] = {}

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
            previous_session_id = self._session_id
            previous_participant_id = self._participant_id
            if normalized_session_id:
                self._session_id = normalized_session_id
            if normalized_participant_id:
                self._participant_id = normalized_participant_id

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
                    turn.session_id in {self.UNKNOWN_SESSION_ID, self._fallback_session_id}
                    and self._session_id
                    not in {self.UNKNOWN_SESSION_ID, self._fallback_session_id}
                ):
                    turn.session_id = self._session_id
                if (
                    turn.participant_id
                    in {self.UNKNOWN_PARTICIPANT_ID, self._fallback_participant_id}
                    and self._participant_id
                    not in {self.UNKNOWN_PARTICIPANT_ID, self._fallback_participant_id}
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

        await self._on_assistant_text(normalized)

    async def on_speech_created(self, speech_handle: Any) -> None:
        """Attach a done callback to capture assistant text when playout is complete."""
        speech_id = self._normalize_optional_text(getattr(speech_handle, "id", None))
        if speech_id:
            self._pending_speech_ids_for_first_audio.append(speech_id)

        # Try immediate extraction first. Some pipelines do not preserve/trigger
        # done callbacks consistently for long responses.
        assistant_text = self._extract_text_from_chat_items(
            getattr(speech_handle, "chat_items", [])
        )
        if assistant_text:
            await self._on_assistant_text(assistant_text)

        add_done_callback = getattr(speech_handle, "add_done_callback", None)
        if not callable(add_done_callback):
            return

        def _on_done(handle: Any) -> None:
            try:
                done_speech_id = self._normalize_optional_text(getattr(handle, "id", None))
                if done_speech_id:
                    self._discard_pending_speech_id(done_speech_id)
                assistant_text = self._extract_text_from_chat_items(
                    getattr(handle, "chat_items", [])
                )
                if not assistant_text:
                    return
                asyncio.create_task(self._on_assistant_text(assistant_text))
            except Exception:
                # Never break realtime pipeline due to observability hooks.
                return

        try:
            add_done_callback(_on_done)
        except Exception:
            return

    async def on_agent_state_changed(
        self,
        *,
        old_state: str,
        new_state: str,
    ) -> None:
        """Record first assistant audio timestamp when agent enters speaking state."""
        if new_state != "speaking":
            return

        speech_id: Optional[str] = None
        while self._pending_speech_ids_for_first_audio:
            candidate = self._pending_speech_ids_for_first_audio.popleft()
            if candidate not in self._first_audio_monotonic_by_speech:
                speech_id = candidate
                break

        if speech_id is None:
            latest = self._latest_agent_speech_id
            if latest and latest not in self._first_audio_monotonic_by_speech:
                speech_id = latest

        if speech_id:
            self._first_audio_monotonic_by_speech.setdefault(speech_id, monotonic())
            logger.debug(
                "First assistant audio recorded from state transition: speech_id=%s, old_state=%s, new_state=%s",
                speech_id,
                old_state,
                new_state,
            )

    async def _on_assistant_text(self, assistant_text: str) -> None:
        normalized = assistant_text.strip()
        if not normalized:
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

        trace_turn = await self._attach_tts_stage(
            speech_id=speech_id,
            duration=duration,
            fallback_duration=audio_duration,
            ttfb=ttfb,
        )
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
            trace_turn = await self._attach_llm_stage(
                duration=collected_metrics.duration,
                ttft=collected_metrics.ttft,
            )

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
                speech_id=speech_id,
                duration=collected_metrics.duration,
                fallback_duration=collected_metrics.audio_duration,
                ttfb=collected_metrics.ttfb,
            )

        elif isinstance(collected_metrics, metrics.EOUMetrics):
            speech_id = collected_metrics.speech_id
            if speech_id:
                self._speech_end_monotonic_by_speech.setdefault(speech_id, monotonic())
                self._eou_delays[speech_id] = collected_metrics.end_of_utterance_delay
                self._stt_finalization_delays[speech_id] = (
                    collected_metrics.transcription_delay
                )
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
                    duration=collected_metrics.end_of_utterance_delay,
                    transcription_delay=collected_metrics.transcription_delay,
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
            stt_finalization_delay = self._stt_finalization_delays.get(speech_id, 0.0)
            observed_total_latency = self._observed_total_latency_seconds(speech_id)
            turn_metrics.compute_latencies(
                eou_delay,
                stt_finalization_delay=stt_finalization_delay,
                observed_total_latency=observed_total_latency,
            )

            # Publish to data channel
            await self._publish_metrics(turn_metrics)

            # Cleanup
            self._pending_metrics.pop(speech_id, None)
            self._eou_delays.pop(speech_id, None)
            self._stt_finalization_delays.pop(speech_id, None)
            self._speech_end_monotonic_by_speech.pop(speech_id, None)
            self._first_audio_monotonic_by_speech.pop(speech_id, None)
            self._discard_pending_speech_id(speech_id)

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
        stt_finalization_delay = self._stt_finalization_delays.get(speech_id or "", 0.0)
        llm_ttft = turn_metrics.llm.ttft if turn_metrics.llm else 0.0
        tts_ttfb = turn_metrics.tts.ttfb if turn_metrics.tts else 0.0
        baseline_total_latency = eou_delay + stt_finalization_delay + llm_ttft + tts_ttfb
        observed_total_latency = self._observed_total_latency_seconds(speech_id or "")
        total_latency = max(
            baseline_total_latency,
            observed_total_latency if observed_total_latency is not None else 0.0,
        )
        llm_to_tts_handoff_latency = max(total_latency - baseline_total_latency, 0.0)
        return {
            "total_latency": total_latency,
            "eou_delay": eou_delay,
            "stt_finalization_delay": stt_finalization_delay,
            "llm_to_tts_handoff_latency": llm_to_tts_handoff_latency,
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
            turn = self._next_turn_where(lambda candidate: candidate.stt_status != "measured")
            if not turn:
                return None
            if transcript:
                turn.user_transcript = transcript.strip()
            turn.prompt_text = turn.user_transcript
            measured_duration_ms = self._duration_to_ms_or_none(duration, fallback_duration)
            if measured_duration_ms is None:
                turn.stt_duration_ms = None
                if turn.stt_total_latency_ms is None:
                    turn.stt_status = "missing"
            else:
                turn.stt_duration_ms = measured_duration_ms
                turn.stt_status = "measured"
            self._recompute_conversational_latency(turn)
            return turn

    async def _attach_vad_stage(
        self,
        *,
        duration: float,
        transcription_delay: float,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_where(lambda candidate: candidate.vad_duration_ms is None)
            if not turn:
                return None
            eou_delay_ms = self._duration_to_ms(duration, 0.0)
            turn.vad_duration_ms = eou_delay_ms
            turn.stt_finalization_ms = self._duration_to_ms(transcription_delay, 0.0)
            turn.stt_total_latency_ms = eou_delay_ms + (turn.stt_finalization_ms or 0.0)
            if turn.stt_total_latency_ms > 0:
                turn.stt_status = "measured"
                if turn.stt_duration_ms is None:
                    turn.stt_duration_ms = turn.stt_total_latency_ms
            self._recompute_conversational_latency(turn)
            return turn

    async def _attach_llm_stage(self, *, duration: float, ttft: float) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_where(lambda candidate: candidate.llm_duration_ms is None)
            if not turn:
                return None
            turn.prompt_text = turn.user_transcript
            turn.llm_duration_ms = self._duration_to_ms(duration, 0.0)
            turn.llm_total_latency_ms = turn.llm_duration_ms
            turn.llm_ttft_ms = self._duration_to_ms(ttft, 0.0)
            self._recompute_conversational_latency(turn)
            return turn

    async def _attach_tts_stage(
        self,
        *,
        speech_id: str,
        duration: float,
        fallback_duration: float,
        ttfb: float,
    ) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_where(
                lambda candidate: (
                    candidate.llm_duration_ms is not None
                    and candidate.tts_duration_ms is None
                )
            )
            if not turn:
                return None
            turn.tts_duration_ms = self._duration_to_ms(duration, fallback_duration)
            turn.tts_ttfb_ms = self._duration_to_ms(ttfb, 0.0)
            self._recompute_conversational_latency(turn)
            observed_total_latency = self._observed_total_latency_seconds(speech_id)
            if observed_total_latency is not None:
                observed_total_latency_ms = observed_total_latency * 1000.0
                baseline_latency_ms = turn.conversational_latency_ms
                turn.conversational_latency_ms = max(
                    observed_total_latency_ms,
                    baseline_latency_ms if baseline_latency_ms is not None else 0.0,
                )
            turn.llm_to_tts_handoff_ms = self._compute_llm_to_tts_handoff_ms(
                total_latency_ms=turn.conversational_latency_ms,
                vad_duration_ms=turn.vad_duration_ms,
                stt_finalization_ms=turn.stt_finalization_ms,
                llm_ttft_ms=turn.llm_ttft_ms,
                tts_ttfb_ms=turn.tts_ttfb_ms,
            )
            return turn

    def _recompute_conversational_latency(self, turn: TraceTurn) -> None:
        turn.conversational_latency_ms = self._compute_conversational_latency_ms(
            vad_duration_ms=turn.vad_duration_ms,
            stt_finalization_ms=turn.stt_finalization_ms,
            llm_ttft_ms=turn.llm_ttft_ms,
            tts_ttfb_ms=turn.tts_ttfb_ms,
        )

    def _compute_conversational_latency_ms(
        self,
        *,
        vad_duration_ms: Optional[float],
        stt_finalization_ms: Optional[float],
        llm_ttft_ms: Optional[float],
        tts_ttfb_ms: Optional[float],
    ) -> Optional[float]:
        components = (
            vad_duration_ms,
            stt_finalization_ms,
            llm_ttft_ms,
            tts_ttfb_ms,
        )
        if any(component is None for component in components):
            return None
        return sum(component for component in components if component is not None)

    def _compute_llm_to_tts_handoff_ms(
        self,
        *,
        total_latency_ms: Optional[float],
        vad_duration_ms: Optional[float],
        stt_finalization_ms: Optional[float],
        llm_ttft_ms: Optional[float],
        tts_ttfb_ms: Optional[float],
    ) -> Optional[float]:
        if total_latency_ms is None:
            return None
        baseline_ms = self._compute_conversational_latency_ms(
            vad_duration_ms=vad_duration_ms,
            stt_finalization_ms=stt_finalization_ms,
            llm_ttft_ms=llm_ttft_ms,
            tts_ttfb_ms=tts_ttfb_ms,
        )
        if baseline_ms is None:
            return None
        return max(total_latency_ms - baseline_ms, 0.0)

    def _discard_pending_speech_id(self, speech_id: str) -> None:
        if not self._pending_speech_ids_for_first_audio:
            return
        self._pending_speech_ids_for_first_audio = deque(
            pending for pending in self._pending_speech_ids_for_first_audio if pending != speech_id
        )

    def _observed_total_latency_seconds(self, speech_id: str) -> Optional[float]:
        start = self._speech_end_monotonic_by_speech.get(speech_id)
        end = self._first_audio_monotonic_by_speech.get(speech_id)
        if start is None or end is None:
            return None
        if end <= start:
            return None
        return end - start

    async def _attach_assistant_text(self, assistant_text: str) -> Optional[TraceTurn]:
        async with self._trace_lock:
            turn = self._next_turn_where(lambda candidate: not candidate.assistant_text)
            if not turn:
                return None
            turn.assistant_text = assistant_text
            turn.response_text = assistant_text
            return turn

    async def _maybe_finalize_trace_turn(self, trace_turn: Optional[TraceTurn]) -> None:
        if not trace_turn:
            return

        completed_turn: Optional[TraceTurn] = None
        schedule_timeout_for_turn: Optional[str] = None
        async with self._trace_lock:
            if trace_turn not in self._pending_trace_turns:
                return
            if not self._is_trace_turn_complete(trace_turn):
                if self._should_schedule_finalize_timeout(trace_turn):
                    schedule_timeout_for_turn = trace_turn.turn_id
            else:
                completed_turn = self._finalize_trace_turn_locked(trace_turn)

        if schedule_timeout_for_turn:
            self._schedule_finalize_timeout(schedule_timeout_for_turn)
        if not completed_turn:
            return
        if completed_turn:
            self._schedule_trace_emit(completed_turn)

    def _next_turn_where(
        self,
        predicate: Callable[[TraceTurn], bool],
    ) -> Optional[TraceTurn]:
        for turn in self._pending_trace_turns:
            if predicate(turn):
                return turn
        return None

    def _is_trace_turn_complete(self, turn: TraceTurn) -> bool:
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

    def _finalize_trace_turn_locked(
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
            fallback_text = self._best_available_assistant_text_locked(turn)
            if fallback_text:
                turn.assistant_text = fallback_text
                if not turn.response_text:
                    turn.response_text = fallback_text
            else:
                turn.assistant_text_missing = True
                unavailable_text = "[assistant text unavailable]"
                turn.assistant_text = unavailable_text
                if not turn.response_text:
                    turn.response_text = unavailable_text

        self._pending_trace_turns.remove(turn)
        self._cancel_finalize_timeout(turn.turn_id)
        return turn

    def _schedule_finalize_timeout(self, turn_id: str) -> None:
        if turn_id in self._trace_finalize_tasks:
            return
        task = asyncio.create_task(self._finalize_trace_turn_after_timeout(turn_id))
        self._trace_finalize_tasks[turn_id] = task
        task.add_done_callback(lambda _: self._trace_finalize_tasks.pop(turn_id, None))

    def _cancel_finalize_timeout(self, turn_id: str) -> None:
        task = self._trace_finalize_tasks.pop(turn_id, None)
        current_task = asyncio.current_task()
        if task and not task.done() and task is not current_task:
            task.cancel()

    async def _finalize_trace_turn_after_timeout(self, turn_id: str) -> None:
        await asyncio.sleep(self._trace_finalize_timeout_sec)

        completed_turn: Optional[TraceTurn] = None
        async with self._trace_lock:
            pending_turn = next(
                (turn for turn in self._pending_trace_turns if turn.turn_id == turn_id),
                None,
            )
            if not pending_turn:
                return

            if self._is_trace_turn_complete(pending_turn):
                completed_turn = self._finalize_trace_turn_locked(pending_turn)
            elif (
                pending_turn.user_transcript
                and pending_turn.llm_duration_ms is not None
                and pending_turn.tts_duration_ms is not None
            ):
                completed_turn = self._finalize_trace_turn_locked(
                    pending_turn,
                    missing_assistant_fallback=True,
                )

        if completed_turn:
            self._schedule_trace_emit(completed_turn)

    def _schedule_trace_emit(self, turn: TraceTurn) -> None:
        if len(self._trace_emit_tasks) >= self._trace_max_pending_tasks:
            logger.warning(
                "Dropping Langfuse trace for turn_id=%s due to pending trace backlog (%s)",
                turn.turn_id,
                len(self._trace_emit_tasks),
            )
            return

        task = asyncio.create_task(self._emit_and_publish_trace_turn(turn))
        self._trace_emit_tasks.add(task)
        task.add_done_callback(lambda completed: self._trace_emit_tasks.discard(completed))

    async def _emit_and_publish_trace_turn(self, turn: TraceTurn) -> None:
        await self._emit_turn_trace(turn)
        await self._publish_trace_update(turn)

    async def wait_for_pending_trace_tasks(self) -> None:
        tasks = list(self._trace_emit_tasks)
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _emit_turn_trace(self, turn: TraceTurn) -> None:
        if not self._langfuse_enabled:
            return

        try:
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
            trace_output = turn.assistant_text or turn.response_text

            root_context = trace.set_span_in_context(trace.INVALID_SPAN)
            root_start_ns = time_ns()
            cursor_ns = root_start_ns

            turn_span = tracer.start_span(
                "turn",
                context=root_context,
                start_time=root_start_ns,
            )
            try:
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
                turn_span.set_attribute("langfuse.trace.output", trace_output)
                turn_span.set_attribute("langfuse.trace.metadata.room_id", turn.room_id)
                turn_span.set_attribute(
                    "langfuse.trace.metadata.participant_id",
                    turn.participant_id,
                )
                turn_span.set_attribute("langfuse.trace.metadata.turn_id", turn.turn_id)
                turn_span.set_attribute(
                    "langfuse.trace.metadata.assistant_text_missing",
                    turn.assistant_text_missing,
                )
                turn_span.set_attribute("langfuse.trace.metadata.stt_status", turn.stt_status)
                turn_span.set_attribute("duration_ms", self._total_duration_ms(turn))
                if user_input_duration_ms is not None:
                    turn_span.set_attribute("latency_ms.user_input", user_input_duration_ms)
                turn_span.set_attribute("latency_ms.vad", vad_duration_ms)
                turn_span.set_attribute("latency_ms.eou_delay", vad_duration_ms)
                if stt_span_duration_ms is not None:
                    turn_span.set_attribute("latency_ms.stt", stt_span_duration_ms)
                if stt_processing_ms is not None:
                    turn_span.set_attribute("latency_ms.stt_processing", stt_processing_ms)
                if stt_finalization_ms is not None:
                    turn_span.set_attribute("latency_ms.stt_finalization", stt_finalization_ms)
                if stt_total_latency_ms is not None:
                    turn_span.set_attribute("latency_ms.stt_total", stt_total_latency_ms)
                turn_span.set_attribute("latency_ms.llm", llm_duration_ms)
                turn_span.set_attribute("latency_ms.llm_ttft", llm_ttft_ms)
                turn_span.set_attribute("latency_ms.llm_total", llm_total_latency_ms)
                turn_span.set_attribute("latency_ms.tts", tts_duration_ms)
                turn_span.set_attribute("latency_ms.tts_ttfb", tts_ttfb_ms)
                if llm_to_tts_handoff_ms is not None:
                    turn_span.set_attribute(
                        "latency_ms.llm_to_tts_handoff",
                        llm_to_tts_handoff_ms,
                    )
                if conversational_latency_ms is not None:
                    turn_span.set_attribute("latency_ms.conversational", conversational_latency_ms)
                    turn_span.set_attribute(
                        "latency_ms.speech_end_to_assistant_speech_start",
                        conversational_latency_ms,
                    )
                turn_span.set_attribute("stt_status", turn.stt_status)

                component_context = trace.set_span_in_context(turn_span)

                cursor_ns = self._emit_component_span(
                    name="user_input",
                    context=component_context,
                    start_ns=cursor_ns,
                    duration_ms=user_input_duration_ms,
                    attributes={"user_transcript": turn.user_transcript},
                    observation_input=turn.user_transcript,
                )
                vad_start_ns = cursor_ns
                cursor_ns = self._emit_component_span(
                    name="vad",
                    context=component_context,
                    start_ns=cursor_ns,
                    duration_ms=vad_duration_ms,
                    attributes={"eou_delay_ms": vad_duration_ms},
                    observation_output=str(vad_duration_ms),
                )
                stt_end_ns = self._emit_component_span(
                    name="stt",
                    context=component_context,
                    start_ns=vad_start_ns,
                    duration_ms=stt_span_duration_ms,
                    attributes={
                        "user_transcript": turn.user_transcript,
                        "stt_status": turn.stt_status,
                        "stt_processing_ms": stt_processing_ms,
                        "stt_finalization_ms": stt_finalization_ms,
                        "stt_total_latency_ms": stt_total_latency_ms,
                    },
                    observation_output=turn.user_transcript,
                )
                cursor_ns = max(cursor_ns, stt_end_ns)
                cursor_ns = self._emit_component_span(
                    name="llm",
                    context=component_context,
                    start_ns=cursor_ns,
                    duration_ms=llm_duration_ms,
                    attributes={
                        "prompt_text": turn.prompt_text,
                        "response_text": turn.response_text,
                        "ttft_ms": llm_ttft_ms,
                        "llm_total_latency_ms": llm_total_latency_ms,
                    },
                    observation_input=turn.prompt_text,
                    observation_output=turn.response_text,
                )
                cursor_ns = self._emit_component_span(
                    name="tts",
                    context=component_context,
                    start_ns=cursor_ns,
                    duration_ms=tts_duration_ms,
                    attributes={
                        "assistant_text": turn.assistant_text,
                        "assistant_text_missing": turn.assistant_text_missing,
                        "ttfb_ms": tts_ttfb_ms,
                    },
                    observation_input=turn.assistant_text,
                    observation_output=turn.assistant_text,
                )
                if conversational_latency_ms is not None:
                    self._emit_component_span(
                        name="conversation_latency",
                        context=component_context,
                        start_ns=vad_start_ns,
                        duration_ms=conversational_latency_ms,
                        attributes={
                            "speech_end_to_assistant_speech_start_ms": conversational_latency_ms,
                            "eou_delay_ms": vad_duration_ms,
                            "stt_finalization_ms": stt_finalization_ms,
                            "llm_ttft_ms": llm_ttft_ms,
                            "llm_to_tts_handoff_ms": llm_to_tts_handoff_ms,
                            "tts_ttfb_ms": tts_ttfb_ms,
                        },
                        observation_output=str(conversational_latency_ms),
                    )
                if llm_to_tts_handoff_ms is not None and llm_to_tts_handoff_ms > 0:
                    llm_to_tts_handoff_start_ns = vad_start_ns + self._duration_ms_to_ns(
                        max(vad_duration_ms, 0.0)
                        + max(stt_finalization_ms or 0.0, 0.0)
                        + max(llm_ttft_ms, 0.0)
                    )
                    self._emit_component_span(
                        name="llm_to_tts_handoff",
                        context=component_context,
                        start_ns=llm_to_tts_handoff_start_ns,
                        duration_ms=llm_to_tts_handoff_ms,
                        attributes={
                            "llm_to_tts_handoff_ms": llm_to_tts_handoff_ms,
                            "speech_end_to_assistant_speech_start_ms": conversational_latency_ms,
                            "eou_delay_ms": vad_duration_ms,
                            "stt_finalization_ms": stt_finalization_ms,
                            "llm_ttft_ms": llm_ttft_ms,
                            "tts_ttfb_ms": tts_ttfb_ms,
                        },
                        observation_output=str(llm_to_tts_handoff_ms),
                    )
            finally:
                turn_total_duration_ns = self._duration_ms_to_ns(self._total_duration_ms(turn))
                self._close_span_at(turn_span, max(cursor_ns, root_start_ns + turn_total_duration_ns))
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
        duration_ms: Optional[float],
        attributes: dict[str, Any],
        observation_input: Optional[str] = None,
        observation_output: Optional[str] = None,
    ) -> int:
        actual_duration_ms = max(duration_ms, 0.0) if duration_ms is not None else None
        end_ns = start_ns + self._duration_ms_to_ns(actual_duration_ms or 0.0)

        span = tracer.start_span(name, context=context, start_time=start_ns)
        try:
            if actual_duration_ms is not None:
                span.set_attribute("duration_ms", actual_duration_ms)
            if observation_input is not None:
                span.set_attribute("input", observation_input)
                span.set_attribute("langfuse.observation.input", observation_input)
            if observation_output is not None:
                span.set_attribute("output", observation_output)
                span.set_attribute("langfuse.observation.output", observation_output)
            for key, value in attributes.items():
                if value is None:
                    continue
                span.set_attribute(key, value)
        finally:
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

    def _duration_ms_to_ns(self, duration_ms: float) -> int:
        return int(max(duration_ms, 0.0) * 1_000_000)

    async def _flush_tracer_provider(self) -> None:
        """Best-effort flush so traces are visible in Langfuse faster."""
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
        stt_component = (
            turn.stt_finalization_ms
            if turn.stt_finalization_ms is not None
            else (turn.stt_duration_ms if turn.stt_duration_ms is not None else 0.0)
        )
        llm_component = (
            turn.llm_total_latency_ms
            if turn.llm_total_latency_ms is not None
            else (turn.llm_duration_ms or 0.0)
        )
        calculated = (
            (turn.vad_duration_ms or 0.0)
            + stt_component
            + llm_component
            + (turn.tts_duration_ms or 0.0)
        )
        if turn.conversational_latency_ms is not None:
            calculated = max(calculated, turn.conversational_latency_ms)
        return calculated

    def _duration_to_ms(self, duration: float, fallback_duration: float) -> float:
        chosen = duration if duration > 0 else fallback_duration
        return max(chosen, 0.0) * 1000.0

    def _duration_to_ms_or_none(self, duration: float, fallback_duration: float) -> Optional[float]:
        chosen = duration if duration > 0 else fallback_duration
        if chosen <= 0:
            return None
        return chosen * 1000.0

    def _normalize_optional_text(self, value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def _build_fallback_id(self, prefix: Optional[str]) -> Optional[str]:
        normalized_prefix = self._normalize_optional_text(prefix)
        if not normalized_prefix:
            return None
        return f"{normalized_prefix}_{uuid.uuid4()}"

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

    def _best_available_assistant_text_locked(self, turn: TraceTurn) -> str:
        if turn.assistant_text.strip():
            return turn.assistant_text.strip()
        if turn.response_text.strip():
            return turn.response_text.strip()
        if self._pending_agent_transcripts:
            return self._pending_agent_transcripts.popleft().strip()
        return ""

    def _extract_content_text(self, content: Any) -> str:
        """Extract plain text from a chat content payload."""
        parts = self._extract_text_parts(content, depth=0, seen=set())
        return " ".join(part.strip() for part in parts if part.strip())

    def _extract_text_parts(
        self,
        value: Any,
        *,
        depth: int,
        seen: set[int],
    ) -> list[str]:
        if value is None or depth > 8:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (bytes, bytearray)):
            return [value.decode("utf-8", errors="ignore")]

        value_id = id(value)
        if value_id in seen:
            return []

        if isinstance(value, dict):
            seen.add(value_id)
            parts: list[str] = []
            preferred_keys = (
                "text",
                "content",
                "parts",
                "message",
                "output",
                "value",
                "delta",
            )
            for key in preferred_keys:
                if key in value:
                    parts.extend(
                        self._extract_text_parts(
                            value[key],
                            depth=depth + 1,
                            seen=seen,
                        )
                    )
            if parts:
                return parts
            for nested in value.values():
                parts.extend(
                    self._extract_text_parts(
                        nested,
                        depth=depth + 1,
                        seen=seen,
                    )
                )
            return parts

        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            seen.add(value_id)
            parts: list[str] = []
            for item in value:
                parts.extend(
                    self._extract_text_parts(
                        item,
                        depth=depth + 1,
                        seen=seen,
                    )
                )
            return parts

        text_attr = getattr(value, "text", None)
        if isinstance(text_attr, str):
            return [text_attr]
        if callable(text_attr):
            text_value = self._safe_text_call(text_attr)
            if text_value is not None:
                parts = self._extract_text_parts(
                    text_value,
                    depth=depth + 1,
                    seen=seen,
                )
                if parts:
                    return parts

        seen.add(value_id)
        parts: list[str] = []
        for attr_name in ("content", "parts", "message", "output", "delta"):
            nested_value = getattr(value, attr_name, None)
            if nested_value is None:
                continue
            parts.extend(
                self._extract_text_parts(
                    nested_value,
                    depth=depth + 1,
                    seen=seen,
                )
            )
        return parts

    def _safe_text_call(self, text_callable: Any) -> Any:
        try:
            return text_callable()
        except TypeError:
            return None
        except Exception:
            return None

    def _extract_text_from_chat_items(self, chat_items: Any) -> str:
        """Extract assistant text from speech handle chat items."""
        if isinstance(chat_items, (str, bytes, bytearray)) or not isinstance(chat_items, Sequence):
            return ""

        parts: list[str] = []
        for item in chat_items:
            role = getattr(item, "role", None)
            if isinstance(role, str) and role not in {"assistant"}:
                continue

            content = getattr(item, "content", None)
            text = self._extract_content_text(content)
            if text.strip():
                parts.append(text.strip())

        # Keep only the last assistant text to best match the turn output.
        if not parts:
            return ""
        return parts[-1]

    def _append_if_new(self, queue: deque[str], value: str) -> None:
        """Avoid duplicate adjacent transcript entries."""
        if queue and queue[-1] == value:
            return
        queue.append(value)
