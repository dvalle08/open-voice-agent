"""Metrics collector for LiveKit agent telemetry.

Aggregates metrics from AgentSession events and publishes to data channel for real-time monitoring.
"""

import json
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from time import time
from typing import Any, Optional, Union

from livekit import rtc
from livekit.agents import metrics

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


class MetricsCollector:
    """Collects and publishes agent metrics to LiveKit data channel."""

    def __init__(self, room: rtc.Room, model_name: str) -> None:
        """Initialize metrics collector.

        Args:
            room: LiveKit room for publishing metrics
            model_name: STT model name to include in metrics
        """
        self._room = room
        self._model_name = model_name
        self._pending_metrics: dict[str, TurnMetrics] = {}
        self._eou_delays: dict[str, float] = {}
        self._pending_transcripts: deque[str] = deque()
        self._pending_agent_transcripts: deque[str] = deque()
        self._latest_agent_speech_id: Optional[str] = None

    async def on_user_input_transcribed(
        self,
        transcript: str,
        *,
        is_final: bool,
    ) -> None:
        """Store final user transcripts for the next STT metrics event."""
        if not is_final:
            return
        normalized = transcript.strip()
        if not normalized:
            return
        self._pending_transcripts.append(normalized)

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
