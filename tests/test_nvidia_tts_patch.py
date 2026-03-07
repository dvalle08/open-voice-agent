from __future__ import annotations

import asyncio
import math
import threading
import time
from typing import Any

from livekit.plugins import nvidia

from src.agent.models import tts_factory


class _FakeResponse:
    def __init__(self, audio: bytes) -> None:
        self.audio = audio


class _FakeSynthesisService:
    def __init__(
        self,
        *,
        chunks: list[bytes] | None = None,
        delay_sec: float = 0.0,
        started_event: threading.Event | None = None,
        finished_event: threading.Event | None = None,
    ) -> None:
        self._chunks = chunks or []
        self._delay_sec = delay_sec
        self._started_event = started_event
        self._finished_event = finished_event

    def synthesize_online(self, *args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs
        if self._started_event is not None:
            self._started_event.set()
        try:
            if self._delay_sec > 0:
                time.sleep(self._delay_sec)
            for chunk in self._chunks:
                yield _FakeResponse(chunk)
        finally:
            if self._finished_event is not None:
                self._finished_event.set()


def test_patched_nvidia_stream_emits_tts_metrics() -> None:
    tts_factory._patch_nvidia_tts_stream_once()
    collected_metrics: list[Any] = []

    async def _run() -> int:
        tts_engine = nvidia.TTS(use_ssl=False)
        tts_engine._ensure_session = lambda: _FakeSynthesisService(
            chunks=[b"\0\0" * 1600, b"\0\0" * 1600]
        )
        tts_engine.on("metrics_collected", lambda metric: collected_metrics.append(metric))

        stream = tts_engine.stream()
        stream.push_text("hello world")
        stream.end_input()

        frames = 0
        async for _ in stream:
            frames += 1
        return frames

    frame_count = asyncio.run(_run())

    assert frame_count > 0
    assert len(collected_metrics) == 1
    metric = collected_metrics[0]
    assert math.isfinite(metric.ttfb)
    assert metric.ttfb >= 0
    assert metric.characters_count == len("hello world")


def test_patched_nvidia_stream_avoids_invalid_state_on_shutdown() -> None:
    tts_factory._patch_nvidia_tts_stream_once()
    loop_errors: list[dict[str, Any]] = []
    started_event = threading.Event()
    finished_event = threading.Event()

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()

        def _capture_exception(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
            loop_errors.append(context)

        loop.set_exception_handler(_capture_exception)
        try:
            tts_engine = nvidia.TTS(use_ssl=False)
            tts_engine._ensure_session = lambda: _FakeSynthesisService(
                delay_sec=0.05,
                started_event=started_event,
                finished_event=finished_event,
            )

            stream = tts_engine.stream()
            stream.push_text("cancel me")
            stream.end_input()

            await asyncio.to_thread(started_event.wait, 1.0)
            await stream.aclose()
            await asyncio.to_thread(finished_event.wait, 1.0)
            await asyncio.sleep(0.05)
        finally:
            loop.set_exception_handler(previous_handler)

    asyncio.run(_run())

    invalid_state_errors = [
        context
        for context in loop_errors
        if isinstance(context.get("exception"), asyncio.InvalidStateError)
    ]
    assert not invalid_state_errors
