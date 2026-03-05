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
from time import monotonic, time
from typing import Any, Optional, Sequence, Union

from livekit import rtc
from livekit.agents import metrics
from livekit.agents.telemetry import tracer  # noqa: F811 – kept at module level for monkeypatch
from opentelemetry import trace  # noqa: F401

from src.agent.traces.channel_metrics import ChannelPublisher
from src.agent.traces.turn_tracer import TraceTurn, TurnTracer
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
        self._pending_transcripts: deque[str] = deque()
        self._pending_agent_transcripts: deque[str] = deque()
        self._pending_speech_ids_for_first_audio: deque[str] = deque()
        self._latest_agent_speech_id: Optional[str] = None
        self._turns: dict[str, TurnState] = {}
        self._pending_llm_watchdog_ids: deque[str] = deque()
        self._llm_stall_tasks: dict[str, asyncio.Task[None]] = {}
        self._latest_vad_metrics: Optional[VADMetrics] = None
        self._latest_vad_metric_attributes: Optional[dict[str, Any]] = None
        self._first_final_user_turn_logged = False
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

        self._tracer = TurnTracer(
            publisher=self._publisher,
            room_id=self._room_id,
            session_id=session_id,
            participant_id=participant_id_resolved,
            fallback_session_id=fallback_session_id,
            fallback_participant_id=fallback_participant_id,
            langfuse_enabled=langfuse_enabled,
            pending_agent_transcripts=self._pending_agent_transcripts,
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

    # ------------------------------------------------------------------
    # Public event handlers
    # ------------------------------------------------------------------

    async def on_session_metadata(
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
        if not is_final:
            return
        normalized = transcript.strip()
        if not normalized:
            return
        self._pending_transcripts.append(normalized)
        if not self._first_final_user_turn_logged:
            self._first_final_user_turn_logged = True
            logger.info(
                "First finalized user transcript received: room=%s chars=%s preview=%r",
                self._room_name,
                len(normalized),
                normalized[:80],
            )
        self._start_llm_stall_watchdog(transcript=normalized)
        room_id = await self._resolve_room_id()
        await self._tracer.create_turn(user_transcript=normalized, room_id=room_id)

    async def on_conversation_item_added(
        self,
        *,
        role: Optional[str],
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
            _append_if_new(self._pending_transcripts, normalized)
            return
        assistant_event_created_at = (
            item_created_at if item_created_at is not None else event_created_at
        )
        await self._on_assistant_text(
            normalized,
            event_created_at=assistant_event_created_at,
        )

    async def on_function_tools_executed(
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
        await self._tracer.maybe_finalize(trace_turn)

    async def on_tool_step_started(self) -> bool:
        trace_turn, should_announce = await self._tracer.attach_tool_step_started()
        await self._tracer.maybe_finalize(trace_turn)
        return should_announce

    async def on_speech_created(self, speech_handle: Any) -> None:
        speech_id = _normalize(getattr(speech_handle, "id", None))
        if speech_id:
            self._pending_speech_ids_for_first_audio.append(speech_id)

        assistant_text, assistant_created_at = _extract_latest_assistant_chat_item(
            getattr(speech_handle, "chat_items", [])
        )
        if assistant_text:
            await self._on_assistant_text(
                assistant_text,
                event_created_at=assistant_created_at,
            )

        add_done_callback = getattr(speech_handle, "add_done_callback", None)
        if not callable(add_done_callback):
            return

        def _on_done(handle: Any) -> None:
            try:
                done_speech_id = _normalize(getattr(handle, "id", None))
                if done_speech_id:
                    self._discard_pending_speech_id(done_speech_id)
                text, created_at = _extract_latest_assistant_chat_item(
                    getattr(handle, "chat_items", [])
                )
                if text:
                    asyncio.create_task(
                        self._on_assistant_text(
                            text,
                            event_created_at=created_at,
                        )
                    )
            except Exception:
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
        if new_state != "speaking":
            return

        speech_id: Optional[str] = None
        while self._pending_speech_ids_for_first_audio:
            candidate = self._pending_speech_ids_for_first_audio.popleft()
            state = self._turns.get(candidate)
            if state is None or state.first_audio_monotonic is None:
                speech_id = candidate
                break

        if speech_id is None:
            latest = self._latest_agent_speech_id
            if latest:
                state = self._turns.get(latest)
                if state is None or state.first_audio_monotonic is None:
                    speech_id = latest

        if speech_id:
            ts = self._get_or_create_state(speech_id)
            if ts.first_audio_monotonic is None:
                ts.first_audio_monotonic = monotonic()
            logger.debug(
                "First assistant audio recorded from state transition: speech_id=%s, old_state=%s, new_state=%s",
                speech_id,
                old_state,
                new_state,
            )

    async def on_tts_synthesized(
        self,
        *,
        ttfb: float,
        duration: float,
        audio_duration: float,
    ) -> None:
        if ttfb < 0:
            return
        speech_id = self._latest_agent_speech_id or f"tts-{uuid.uuid4()}"
        turn_metrics = self._get_or_create_turn(speech_id, role="agent")
        turn_metrics.tts = TTSMetrics(
            type="tts_metrics",
            label="tts_fallback",
            request_id=f"fallback-{speech_id}",
            timestamp=time(),
            duration=duration,
            ttfb=ttfb,
            audio_duration=audio_duration,
            cancelled=False,
            characters_count=0,
            streamed=True,
            speech_id=speech_id,
        )
        await self._publish_live_update(speech_id=speech_id, stage="tts", turn_metrics=turn_metrics)
        logger.debug("TTS fallback metrics collected: speech_id=%s, ttfb=%.3fs", speech_id, ttfb)
        await self._maybe_publish_turn(speech_id, turn_metrics)

        trace_turn = await self._tracer.attach_tts(
            duration=duration,
            fallback_duration=audio_duration,
            ttfb=ttfb,
            speech_id=speech_id,
            observed_total_latency=self._observed_total_latency(speech_id),
            metric_attributes={
                "type": "tts_metrics",
                "label": "tts_fallback",
                "request_id": f"fallback-{speech_id}",
                "timestamp": time(),
                "duration": duration,
                "ttfb": ttfb,
                "audio_duration": audio_duration,
                "cancelled": False,
                "characters_count": 0,
                "streamed": True,
                "speech_id": speech_id,
            },
        )
        await self._tracer.maybe_finalize(trace_turn)

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
        speech_id = None
        turn_metrics = None
        trace_turn: Optional[TraceTurn] = None

        if isinstance(collected_metrics, metrics.STTMetrics):
            speech_id = collected_metrics.request_id
            turn_metrics = self._get_or_create_turn(speech_id, role="user")
            if self._pending_transcripts:
                turn_metrics.transcript = self._pending_transcripts.popleft()
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
            self._mark_llm_stage_reached()
            speech_id = collected_metrics.speech_id or collected_metrics.request_id
            turn_metrics = self._get_or_create_turn(speech_id, role="agent")
            self._latest_agent_speech_id = speech_id
            if self._pending_agent_transcripts and not turn_metrics.transcript:
                turn_metrics.transcript = self._pending_agent_transcripts.popleft()
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
            turn_metrics = self._get_or_create_turn(speech_id, role="agent")
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
                metadata=_metric_metadata_to_dict(collected_metrics.metadata),
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

        elif isinstance(collected_metrics, metrics.EOUMetrics):
            speech_id = collected_metrics.speech_id
            if speech_id:
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
        await self._tracer.maybe_finalize(trace_turn)

    async def wait_for_pending_trace_tasks(self) -> None:
        await self._tracer.wait_for_pending_tasks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
    ) -> None:
        normalized = assistant_text.strip()
        if not normalized:
            return
        _append_if_new(self._pending_agent_transcripts, normalized)
        trace_turn = await self._tracer.attach_assistant_text(
            normalized,
            event_created_at=event_created_at,
        )
        await self._tracer.maybe_finalize(trace_turn)

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
        self._discard_pending_speech_id(speech_id)

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

    def _discard_pending_speech_id(self, speech_id: str) -> None:
        if not self._pending_speech_ids_for_first_audio:
            return
        self._pending_speech_ids_for_first_audio = deque(
            s for s in self._pending_speech_ids_for_first_audio if s != speech_id
        )

    def _start_llm_stall_watchdog(self, *, transcript: str) -> None:
        if self._llm_stall_timeout_sec <= 0:
            return

        watchdog_id = str(uuid.uuid4())
        self._pending_llm_watchdog_ids.append(watchdog_id)
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

    def _mark_llm_stage_reached(self) -> None:
        while self._pending_llm_watchdog_ids:
            watchdog_id = self._pending_llm_watchdog_ids.popleft()
            task = self._llm_stall_tasks.pop(watchdog_id, None)
            if task:
                task.cancel()
                return

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

        if watchdog_id not in self._pending_llm_watchdog_ids:
            return

        self._pending_llm_watchdog_ids = deque(
            wid for wid in self._pending_llm_watchdog_ids if wid != watchdog_id
        )

        preview = transcript[:80]
        logger.warning(
            "Turn stalled before LLM stage: timeout=%.2fs room=%s transcript_chars=%s transcript_preview=%r",
            self._llm_stall_timeout_sec,
            self._room_name,
            len(transcript),
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
