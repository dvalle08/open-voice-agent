from __future__ import annotations

import asyncio
import random
from typing import Any

from src.agent.tool_feedback import ToolFeedbackController


class _FakePlayHandle:
    def __init__(self) -> None:
        self._done = False
        self.stop_calls = 0

    def done(self) -> bool:
        return self._done

    def stop(self) -> None:
        self.stop_calls += 1
        self._done = True


class _FakeBackgroundAudioPlayer:
    def __init__(self) -> None:
        self.start_calls = 0
        self.play_calls: list[tuple[Any, bool]] = []
        self.closed = False
        self.last_handle: _FakePlayHandle | None = None

    async def start(self, *, room: Any, agent_session: Any) -> None:
        self.start_calls += 1

    def play(self, clip: Any, *, loop: bool = False) -> _FakePlayHandle:
        self.play_calls.append((clip, loop))
        handle = _FakePlayHandle()
        self.last_handle = handle
        return handle

    async def aclose(self) -> None:
        self.closed = True


def test_tool_feedback_rotates_fallback_phrases_without_repeats_in_one_cycle() -> None:
    phrases = ("one", "two", "three")
    controller = ToolFeedbackController(
        enabled=False,
        fallback_phrases=phrases,
        rng=random.Random(7),
    )

    first_cycle = [controller.next_fallback_phrase() for _ in range(len(phrases))]
    assert sorted(first_cycle) == sorted(phrases)


def test_tool_feedback_typing_sound_lifecycle() -> None:
    async def _run() -> _FakeBackgroundAudioPlayer:
        fake_player = _FakeBackgroundAudioPlayer()
        controller = ToolFeedbackController(
            enabled=True,
            audio_player_factory=lambda: fake_player,
            typing_timeout_sec=1.0,
            rng=random.Random(1),
        )
        await controller.start(room=object(), session=object())  # type: ignore[arg-type]
        await controller.start_typing_sound()
        await controller.start_typing_sound()
        await controller.stop_typing_sound(reason="test")
        await controller.aclose()
        return fake_player

    fake_player = asyncio.run(_run())
    assert fake_player.start_calls == 1
    assert len(fake_player.play_calls) == 1
    assert fake_player.last_handle is not None
    assert fake_player.last_handle.stop_calls == 1
    assert fake_player.closed is True


def test_tool_feedback_typing_sound_auto_stops_on_timeout() -> None:
    async def _run() -> _FakeBackgroundAudioPlayer:
        fake_player = _FakeBackgroundAudioPlayer()
        controller = ToolFeedbackController(
            enabled=True,
            audio_player_factory=lambda: fake_player,
            typing_timeout_sec=0.01,
            rng=random.Random(1),
        )
        await controller.start(room=object(), session=object())  # type: ignore[arg-type]
        await controller.start_typing_sound()
        await asyncio.sleep(0.05)
        await controller.aclose()
        return fake_player

    fake_player = asyncio.run(_run())
    assert fake_player.last_handle is not None
    assert fake_player.last_handle.stop_calls == 1
