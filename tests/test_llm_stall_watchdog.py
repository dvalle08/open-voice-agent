from __future__ import annotations

import asyncio
from typing import Any

from livekit.agents import metrics

from src.agent.traces.metrics_collector import MetricsCollector


class _AwaitableValue:
    def __init__(self, value: str) -> None:
        self._value = value

    def __await__(self):  # type: ignore[no-untyped-def]
        async def _inner() -> str:
            return self._value

        return _inner().__await__()


class _FakeLocalParticipant:
    async def publish_data(
        self,
        *,
        payload: bytes,
        topic: str,
        reliable: bool,
    ) -> None:
        _ = payload, topic, reliable


class _FakeRoom:
    def __init__(self) -> None:
        self.name = "voice-stall-test"
        self.sid = _AwaitableValue("RM_STALL")
        self.local_participant = _FakeLocalParticipant()


def _make_llm_metrics(speech_id: str) -> metrics.LLMMetrics:
    return metrics.LLMMetrics(
        label="llm",
        request_id=f"req-{speech_id}",
        timestamp=0.0,
        duration=0.5,
        ttft=0.1,
        cancelled=False,
        completion_tokens=24,
        prompt_tokens=12,
        prompt_cached_tokens=0,
        total_tokens=36,
        tokens_per_second=40.0,
        speech_id=speech_id,
    )


def test_turn_stall_watchdog_logs_warning_when_llm_stage_never_arrives(
    caplog: Any,
) -> None:
    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="nvidia",
        room_name=room.name,
        room_id="RM_STALL",
        participant_id="web-stall",
        langfuse_enabled=False,
    )
    collector._llm_stall_timeout_sec = 0.01
    caplog.set_level("WARNING", logger="open_voice_agent")

    async def _run() -> None:
        await collector.on_user_input_transcribed("hello agent", is_final=True)
        await asyncio.sleep(0.02)

    asyncio.run(_run())

    assert any(
        "Turn stalled before LLM stage" in record.getMessage()
        for record in caplog.records
    )


def test_turn_stall_watchdog_is_cleared_once_llm_metrics_arrive(
    caplog: Any,
) -> None:
    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="nvidia",
        room_name=room.name,
        room_id="RM_STALL",
        participant_id="web-stall",
        langfuse_enabled=False,
    )
    collector._llm_stall_timeout_sec = 0.05
    caplog.set_level("WARNING", logger="open_voice_agent")

    async def _run() -> None:
        await collector.on_user_input_transcribed("hello agent", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-1"))
        await asyncio.sleep(0.06)

    asyncio.run(_run())

    assert not any(
        "Turn stalled before LLM stage" in record.getMessage()
        for record in caplog.records
    )
