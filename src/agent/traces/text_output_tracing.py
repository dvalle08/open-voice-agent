"""Tracing helpers for generic LiveKit text output capture."""

from __future__ import annotations

import asyncio
from collections import deque
from time import time
from typing import Any, Callable, Optional

from livekit.agents.voice import io as voice_io

from src.core.logger import logger

try:
    from livekit.agents.voice.agent_activity import _SpeechHandleContextVar
except Exception:  # pragma: no cover - private upstream hook may disappear
    _SpeechHandleContextVar = None


class TracingTextOutput(voice_io.TextOutput):
    """Wraps the session transcription sink to capture assistant text by speech id."""

    def __init__(
        self,
        *,
        on_delta: Callable[[Optional[str], str, float], None],
        on_flush: Callable[[Optional[str], float], None],
        on_context_missing: Callable[[float], None],
        next_in_chain: voice_io.TextOutput | None,
    ) -> None:
        super().__init__(label="TracingTextOutput", next_in_chain=next_in_chain)
        self._next_in_chain = next_in_chain
        self._on_delta = on_delta
        self._on_flush = on_flush
        self._on_context_missing = on_context_missing
        self._task_speech_ids: dict[asyncio.Task[Any], str] = {}
        self._missing_context_var_logged = False
        self._missing_speech_context_logged = False

    async def capture_text(self, text: str) -> None:
        observed_at = time()
        speech_id = self._resolve_current_speech_id()
        current_task = asyncio.current_task()
        if speech_id and current_task is not None:
            self._task_speech_ids[current_task] = speech_id
        if text:
            self._safe_on_delta(speech_id=speech_id, text=text, observed_at=observed_at)
        if speech_id is None and text.strip():
            self._log_missing_speech_context()
            self._safe_on_context_missing(observed_at)

        if self._next_in_chain is not None:
            await self._next_in_chain.capture_text(text)

    def flush(self) -> None:
        observed_at = time()
        current_task = asyncio.current_task()
        speech_id: Optional[str] = None
        if current_task is not None:
            speech_id = self._task_speech_ids.pop(current_task, None)
        if speech_id is None:
            speech_id = self._resolve_current_speech_id()

        try:
            if self._next_in_chain is not None:
                self._next_in_chain.flush()
        finally:
            self._safe_on_flush(speech_id=speech_id, observed_at=observed_at)

    def _resolve_current_speech_id(self) -> Optional[str]:
        if _SpeechHandleContextVar is None:
            if not self._missing_context_var_logged:
                self._missing_context_var_logged = True
                logger.warning(
                    "LiveKit SpeechHandle context unavailable; streamed assistant tracing will degrade"
                )
            return None

        speech_handle = _SpeechHandleContextVar.get(None)
        if speech_handle is None:
            return None
        speech_id = getattr(speech_handle, "id", None)
        if not isinstance(speech_id, str):
            return None
        normalized = speech_id.strip()
        return normalized or None

    def _safe_on_delta(
        self,
        *,
        speech_id: Optional[str],
        text: str,
        observed_at: float,
    ) -> None:
        try:
            self._on_delta(speech_id, text, observed_at)
        except Exception:
            logger.exception(
                "Failed to submit streamed assistant text delta for tracing: speech_id=%s",
                speech_id,
            )

    def _safe_on_flush(
        self,
        *,
        speech_id: Optional[str],
        observed_at: float,
    ) -> None:
        try:
            self._on_flush(speech_id, observed_at)
        except Exception:
            logger.exception(
                "Failed to submit streamed assistant text flush for tracing: speech_id=%s",
                speech_id,
            )

    def _safe_on_context_missing(self, observed_at: float) -> None:
        try:
            self._on_context_missing(observed_at)
        except Exception:
            logger.exception("Failed to submit missing streamed assistant speech context")

    def _log_missing_speech_context(self) -> None:
        if self._missing_speech_context_logged:
            return
        self._missing_speech_context_logged = True
        logger.warning(
            "Streamed assistant text arrived without SpeechHandle context; Langfuse tracing may fall back"
        )


def install_tracing_text_output(
    *,
    session: Any,
    on_delta: Callable[[Optional[str], str, float], None],
    on_flush: Callable[[Optional[str], float], None],
    on_context_missing: Callable[[float], None],
) -> TracingTextOutput:
    """Install the tracing text sink ahead of the existing session sink."""

    current_sink = getattr(session.output, "transcription", None)
    if isinstance(current_sink, TracingTextOutput):
        return current_sink

    tracing_sink = TracingTextOutput(
        on_delta=on_delta,
        on_flush=on_flush,
        on_context_missing=on_context_missing,
        next_in_chain=current_sink,
    )
    session.output.transcription = tracing_sink
    return tracing_sink


def recent_unscoped_stream_events(
    event_times: deque[float],
    *,
    since: float | None,
) -> bool:
    """Whether any unscoped stream event happened since the given timestamp."""

    if since is None:
        return bool(event_times)
    return any(event_at >= since for event_at in event_times)
