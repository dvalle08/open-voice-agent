from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from src.agent.traces.channel_metrics import ChannelPublisher
from src.agent.traces.metrics_collector import LLMMetrics, TTSMetrics, TurnMetrics


class _FailingLocalParticipant:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def publish_data(
        self,
        *,
        payload: bytes,
        topic: str,
        reliable: bool,
    ) -> None:
        _ = (payload, topic, reliable)
        raise self._exc


class _FakeRoom:
    def __init__(self, exc: Exception) -> None:
        self.local_participant = _FailingLocalParticipant(exc)


class _CapturingLocalParticipant:
    def __init__(self) -> None:
        self.published_messages: list[dict[str, Any]] = []

    async def publish_data(
        self,
        *,
        payload: bytes,
        topic: str,
        reliable: bool,
    ) -> None:
        self.published_messages.append(
            {
                "payload": json.loads(payload.decode("utf-8")),
                "topic": topic,
                "reliable": reliable,
            }
        )


class _CapturingRoom:
    def __init__(self) -> None:
        self.local_participant = _CapturingLocalParticipant()


def test_publish_live_update_downgrades_preconnect_publish_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.channel_metrics as channel_metrics_module

    debug_logs: list[tuple[Any, ...]] = []
    error_logs: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        channel_metrics_module.logger,
        "debug",
        lambda *args, **kwargs: debug_logs.append((args, kwargs)),
    )
    monkeypatch.setattr(
        channel_metrics_module.logger,
        "error",
        lambda *args, **kwargs: error_logs.append((args, kwargs)),
    )

    room = _FakeRoom(
        RuntimeError("cannot access local participant before connecting")
    )
    publisher = ChannelPublisher(room)  # type: ignore[arg-type]

    asyncio.run(
        publisher.publish_live_update(
            speech_id=None,
            stage="llm",
            role=None,
            turn_metrics=None,
        )
    )

    assert debug_logs
    assert not error_logs


def test_publish_live_update_keeps_error_logging_for_other_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.channel_metrics as channel_metrics_module

    debug_logs: list[tuple[Any, ...]] = []
    error_logs: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        channel_metrics_module.logger,
        "debug",
        lambda *args, **kwargs: debug_logs.append((args, kwargs)),
    )
    monkeypatch.setattr(
        channel_metrics_module.logger,
        "error",
        lambda *args, **kwargs: error_logs.append((args, kwargs)),
    )

    room = _FakeRoom(RuntimeError("unexpected publish failure"))
    publisher = ChannelPublisher(room)  # type: ignore[arg-type]

    asyncio.run(
        publisher.publish_live_update(
            speech_id=None,
            stage="llm",
            role=None,
            turn_metrics=None,
        )
    )

    assert error_logs
    assert not debug_logs


def test_publish_live_update_omits_removed_handoff_latency() -> None:
    room = _CapturingRoom()
    publisher = ChannelPublisher(room)  # type: ignore[arg-type]
    turn_metrics = TurnMetrics(
        turn_id="turn-live-update",
        timestamp=123.0,
        role="agent",
        llm=LLMMetrics(
            type="llm_metrics",
            label="LLM Metrics",
            request_id="llm-123",
            timestamp=123.0,
            duration=0.8,
            ttft=0.2,
            cancelled=False,
            completion_tokens=42,
            prompt_tokens=10,
            prompt_cached_tokens=0,
            total_tokens=52,
            tokens_per_second=52.5,
        ),
        tts=TTSMetrics(
            type="tts_metrics",
            label="TTS Metrics",
            request_id="tts-123",
            timestamp=123.1,
            duration=0.6,
            ttfb=0.3,
            audio_duration=1.2,
            cancelled=False,
            characters_count=17,
            streamed=True,
        ),
    )

    asyncio.run(
        publisher.publish_live_update(
            speech_id="speech-live-update",
            stage="tts",
            role="agent",
            turn_metrics=turn_metrics,
            eou_delay=0.5,
            stt_finalization_delay=0.1,
            observed_total_latency=1.5,
        )
    )

    assert len(room.local_participant.published_messages) == 1
    payload = room.local_participant.published_messages[0]["payload"]
    assert payload["type"] == "metrics_live_update"
    assert payload["latencies"] == {
        "total_latency": pytest.approx(1.5),
        "eou_delay": pytest.approx(0.5),
        "stt_finalization_delay": pytest.approx(0.1),
        "vad_detection_delay": pytest.approx(0.5),
        "llm_ttft": pytest.approx(0.2),
        "tts_ttfb": pytest.approx(0.3),
    }
    assert "llm_to_tts_handoff_latency" not in payload["latencies"]
