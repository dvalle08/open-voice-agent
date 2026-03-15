"""Data channel metrics publisher for LiveKit agent telemetry.

Publishes four message types to the LiveKit data channel:
- ``metrics_live_update``     – ephemeral per-stage updates
- ``conversation_turn``       – completed turn metrics
- ``turn_pipeline_summary``   – finalized pipeline view model for frontend
- ``trace_update``            – Langfuse trace ID notification
"""

from __future__ import annotations

import json
from dataclasses import asdict
from time import time
from typing import TYPE_CHECKING, Any, Optional

from livekit import rtc

from src.core.logger import logger

if TYPE_CHECKING:
    from src.agent.traces.metrics_collector import STTMetrics, TurnMetrics, VADMetrics


class ChannelPublisher:
    """Publishes agent metrics to a LiveKit data channel."""

    def __init__(self, room: rtc.Room) -> None:
        self._room = room

    async def publish_live_update(
        self,
        *,
        speech_id: Optional[str],
        stage: str,
        role: Optional[str],
        turn_metrics: Optional[TurnMetrics],
        vad_metrics: Optional[VADMetrics] = None,
        diagnostic: bool = False,
        eou_delay: float = 0.0,
        stt_finalization_delay: float = 0.0,
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
                    "eou": None,
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
                        "display_duration": _stt_display_duration(turn_metrics.stt),
                    }
                if turn_metrics.eou:
                    payload["metrics"]["eou"] = asdict(turn_metrics.eou)
                if turn_metrics.llm:
                    payload["metrics"]["llm"] = asdict(turn_metrics.llm)
                if turn_metrics.tts:
                    payload["metrics"]["tts"] = asdict(turn_metrics.tts)
                if turn_metrics.vad:
                    payload["metrics"]["vad"] = asdict(turn_metrics.vad)

            payload["latencies"] = _build_partial_latencies(
                turn_metrics=turn_metrics,
                eou_delay=eou_delay,
                stt_finalization_delay=stt_finalization_delay,
            )

            if vad_metrics:
                payload["metrics"]["vad"] = asdict(vad_metrics)

            await self._room.local_participant.publish_data(
                payload=json.dumps(payload).encode("utf-8"),
                topic="metrics",
                reliable=True,
            )
        except Exception as e:
            if _is_preconnect_publish_error(e):
                logger.debug(
                    "Skipping live metrics update before room connect: %s",
                    e,
                )
            else:
                logger.error(f"Failed to publish live metrics update: {e}")

    async def publish_conversation_turn(self, turn_metrics: TurnMetrics) -> None:
        """Publish completed turn metrics to LiveKit data channel."""
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

    async def publish_trace_update(
        self,
        *,
        session_id: str,
        turn_id: str,
        trace_id: str,
    ) -> None:
        """Publish trace update notification to LiveKit data channel."""
        try:
            payload = {
                "type": "trace_update",
                "session_id": session_id,
                "turn_id": turn_id,
                "trace_id": trace_id,
                "timestamp": time(),
            }
            await self._room.local_participant.publish_data(
                payload=json.dumps(payload).encode("utf-8"),
                topic="metrics",
                reliable=True,
            )
        except Exception as exc:
            logger.error(f"Failed to publish trace update: {exc}")

    async def publish_turn_pipeline_summary(self, payload: dict[str, Any]) -> None:
        """Publish finalized turn pipeline summary for frontend rendering."""
        try:
            await self._room.local_participant.publish_data(
                payload=json.dumps(payload).encode("utf-8"),
                topic="metrics",
                reliable=True,
            )
        except Exception as exc:
            logger.error(f"Failed to publish turn pipeline summary: {exc}")


def _stt_display_duration(stt_metrics: STTMetrics) -> float:
    """Prefer measured STT duration, fallback to audio duration if missing."""
    if stt_metrics.duration > 0:
        return stt_metrics.duration
    return stt_metrics.audio_duration


def _is_preconnect_publish_error(exc: Exception) -> bool:
    return "cannot access local participant before connecting" in str(exc).lower()


def _build_partial_latencies(
    *,
    turn_metrics: Optional[TurnMetrics],
    eou_delay: float,
    stt_finalization_delay: float,
) -> Optional[dict[str, float]]:
    """Build a latency preview for live updates."""
    llm_ttft = turn_metrics.llm.ttft if turn_metrics and turn_metrics.llm else 0.0
    tts_ttfb = turn_metrics.tts.ttfb if turn_metrics and turn_metrics.tts else 0.0
    has_signal = any(
        value > 0
        for value in (
            eou_delay,
            stt_finalization_delay,
            llm_ttft,
            tts_ttfb,
        )
    )
    if not has_signal:
        return None

    total = eou_delay + llm_ttft + tts_ttfb
    return {
        "total_latency": total,
        "eou_delay": eou_delay,
        "stt_finalization_delay": stt_finalization_delay,
        "vad_detection_delay": eou_delay,
        "llm_ttft": llm_ttft,
        "tts_ttfb": tts_ttfb,
    }
