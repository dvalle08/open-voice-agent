from __future__ import annotations

import asyncio
import contextvars
from types import SimpleNamespace
from typing import Optional

import pytest
from livekit.agents.voice import io as voice_io

import src.agent.traces.text_output_tracing as text_output_tracing_module
from src.agent.traces.text_output_tracing import install_tracing_text_output


class _FakeNextTextOutput(voice_io.TextOutput):
    def __init__(self) -> None:
        super().__init__(label="FakeNext", next_in_chain=None)
        self.captured: list[str] = []
        self.flush_count = 0

    async def capture_text(self, text: str) -> None:
        self.captured.append(text)

    def flush(self) -> None:
        self.flush_count += 1


def test_tracing_text_output_forwards_and_captures_exact_speech_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_var: contextvars.ContextVar[object | None] = contextvars.ContextVar(
        "speech_handle_ctx",
        default=None,
    )
    monkeypatch.setattr(
        text_output_tracing_module,
        "_SpeechHandleContextVar",
        context_var,
    )

    next_sink = _FakeNextTextOutput()
    session = SimpleNamespace(output=SimpleNamespace(transcription=next_sink))
    deltas: list[tuple[Optional[str], str]] = []
    flushes: list[Optional[str]] = []
    context_missing: list[float] = []

    tracing_sink = install_tracing_text_output(
        session=session,
        on_delta=lambda speech_id, text, observed_at: deltas.append((speech_id, text)),
        on_flush=lambda speech_id, observed_at: flushes.append(speech_id),
        on_context_missing=lambda observed_at: context_missing.append(observed_at),
    )

    async def _run() -> None:
        token = context_var.set(SimpleNamespace(id="speech-stream-test"))
        try:
            await tracing_sink.capture_text("hello")
            tracing_sink.flush()
        finally:
            context_var.reset(token)

    asyncio.run(_run())

    assert session.output.transcription is tracing_sink
    assert deltas == [("speech-stream-test", "hello")]
    assert flushes == ["speech-stream-test"]
    assert context_missing == []
    assert next_sink.captured == ["hello"]
    assert next_sink.flush_count == 1


def test_tracing_text_output_degrades_safely_without_speech_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        text_output_tracing_module,
        "_SpeechHandleContextVar",
        None,
    )

    next_sink = _FakeNextTextOutput()
    session = SimpleNamespace(output=SimpleNamespace(transcription=next_sink))
    deltas: list[tuple[Optional[str], str]] = []
    flushes: list[Optional[str]] = []
    context_missing: list[float] = []

    tracing_sink = install_tracing_text_output(
        session=session,
        on_delta=lambda speech_id, text, observed_at: deltas.append((speech_id, text)),
        on_flush=lambda speech_id, observed_at: flushes.append(speech_id),
        on_context_missing=lambda observed_at: context_missing.append(observed_at),
    )

    async def _run() -> None:
        await tracing_sink.capture_text("hello without context")
        tracing_sink.flush()

    asyncio.run(_run())

    assert deltas == [(None, "hello without context")]
    assert flushes == [None]
    assert len(context_missing) == 1
    assert next_sink.captured == ["hello without context"]
    assert next_sink.flush_count == 1
