from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import Callable, Sequence
from typing import Any

from livekit import rtc
from livekit.agents import AgentSession, BackgroundAudioPlayer, BuiltinAudioClip

from src.agent.prompts.runtime import (
    DEFAULT_TOOL_FALLBACK_PHRASES,
    TOOL_PRE_SPEECH_FALLBACK,
)
from src.core.logger import logger

DEFAULT_TYPING_CLIPS: tuple[BuiltinAudioClip, ...] = (
    BuiltinAudioClip.KEYBOARD_TYPING,
    BuiltinAudioClip.KEYBOARD_TYPING2,
)
DEFAULT_TYPING_TIMEOUT_SEC = 30.0


class ToolFeedbackController:
    """Controls pre-tool fallback speech selection and typing sound playback."""

    def __init__(
        self,
        *,
        enabled: bool,
        fallback_phrases: Sequence[str] = DEFAULT_TOOL_FALLBACK_PHRASES,
        typing_clips: Sequence[BuiltinAudioClip] = DEFAULT_TYPING_CLIPS,
        typing_timeout_sec: float = DEFAULT_TYPING_TIMEOUT_SEC,
        audio_player_factory: Callable[[], Any] = BackgroundAudioPlayer,
        rng: random.Random | None = None,
    ) -> None:
        self._enabled = enabled
        self._fallback_phrases = tuple(p for p in fallback_phrases if p and p.strip())
        self._typing_clips = tuple(typing_clips)
        self._typing_timeout_sec = max(float(typing_timeout_sec), 0.0)
        self._audio_player_factory = audio_player_factory
        self._rng = rng or random.Random()

        self._audio_player: Any | None = None
        self._typing_handle: Any | None = None
        self._typing_timeout_task: asyncio.Task[Any] | None = None

        self._phrase_rotation: list[str] = []
        self._clip_rotation: list[BuiltinAudioClip] = []
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def next_fallback_phrase(self) -> str:
        if not self._fallback_phrases:
            return TOOL_PRE_SPEECH_FALLBACK
        return self._next_rotation_value(
            values=self._fallback_phrases,
            bucket=self._phrase_rotation,
        )

    async def start(self, *, room: rtc.Room, session: AgentSession) -> None:
        if not self._enabled:
            return

        async with self._lock:
            if self._audio_player is not None:
                return
            player = self._audio_player_factory()
            self._audio_player = player

        try:
            await player.start(room=room, agent_session=session)
        except Exception as exc:
            logger.warning("Tool feedback audio unavailable: failed to start background audio: %s", exc)
            async with self._lock:
                self._audio_player = None
            with contextlib.suppress(Exception):
                await player.aclose()
            return

        logger.info("Tool feedback audio started")

    async def start_typing_sound(self) -> None:
        if not self._enabled:
            return

        async with self._lock:
            if self._audio_player is None or not self._typing_clips:
                return

            if self._typing_handle is not None and not self._is_handle_done(self._typing_handle):
                return

            clip = self._next_rotation_value(
                values=self._typing_clips,
                bucket=self._clip_rotation,
            )
            try:
                self._typing_handle = self._audio_player.play(clip, loop=False)
            except Exception as exc:
                logger.warning("Failed to start typing sound: %s", exc)
                return

            self._restart_typing_timeout_task_locked()
            logger.info("typing_sound_started clip=%s", clip.name)

    async def stop_typing_sound(self, *, reason: str) -> None:
        handle: Any | None = None
        async with self._lock:
            timeout_task = self._typing_timeout_task
            self._typing_timeout_task = None
            if timeout_task is not None:
                timeout_task.cancel()

            handle = self._typing_handle
            self._typing_handle = None

        if handle is None:
            return

        try:
            if not self._is_handle_done(handle):
                stop = getattr(handle, "stop", None)
                if callable(stop):
                    stop()
        except Exception as exc:
            logger.warning("Failed to stop typing sound: %s", exc)
        finally:
            logger.info("typing_sound_stopped reason=%s", reason)

    async def aclose(self) -> None:
        await self.stop_typing_sound(reason="shutdown")

        async with self._lock:
            player = self._audio_player
            self._audio_player = None

        if player is not None:
            with contextlib.suppress(Exception):
                await player.aclose()

    def _next_rotation_value(self, *, values: Sequence[Any], bucket: list[Any]) -> Any:
        if not bucket:
            bucket.extend(values)
            self._rng.shuffle(bucket)
        return bucket.pop()

    def _restart_typing_timeout_task_locked(self) -> None:
        if self._typing_timeout_task is not None:
            self._typing_timeout_task.cancel()

        if self._typing_timeout_sec <= 0:
            self._typing_timeout_task = None
            return

        self._typing_timeout_task = asyncio.create_task(
            self._typing_timeout_worker(),
            name="tool-feedback-typing-timeout",
        )

    async def _typing_timeout_worker(self) -> None:
        try:
            await asyncio.sleep(self._typing_timeout_sec)
            await self.stop_typing_sound(reason="timeout")
        except asyncio.CancelledError:
            return

    @staticmethod
    def _is_handle_done(handle: Any) -> bool:
        done = getattr(handle, "done", None)
        if callable(done):
            try:
                return bool(done())
            except Exception:
                return False
        return False
