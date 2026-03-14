from __future__ import annotations

import asyncio
from collections import deque
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Any

import pytest

from opentelemetry import trace as otel_trace
from livekit.agents import metrics

from src.agent.traces.channel_metrics import ChannelPublisher
from src.agent.traces.metrics_collector import MetricsCollector
from src.agent.traces.turn_tracer import (
    PendingAssistantItemRecord,
    PendingUnscopedStreamRecord,
    TurnTracer,
)


@dataclass
class _FakeSpanContext:
    trace_id: int


@dataclass
class _FakeSpan:
    name: str
    trace_id: int
    attributes: dict[str, Any] = field(default_factory=dict)
    end_count: int = 0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def get_span_context(self) -> _FakeSpanContext:
        return _FakeSpanContext(trace_id=self.trace_id)

    def end(self, end_time: Any = None) -> None:
        _ = end_time
        self.end_count += 1


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []
        self._stack: list[_FakeSpan] = []
        self._next_trace_id = 1

    def start_span(self, name: str, **kwargs: Any) -> _FakeSpan:
        trace_id = None
        context = kwargs.get("context")
        if context is not None:
            parent_span = otel_trace.get_current_span(context)
            get_span_context = getattr(parent_span, "get_span_context", None)
            if callable(get_span_context):
                trace_id = get_span_context().trace_id
        if not trace_id:
            trace_id = self._next_trace_id
            self._next_trace_id += 1

        span = _FakeSpan(name=name, trace_id=trace_id)
        self.spans.append(span)
        return span

    @contextmanager
    def start_as_current_span(self, name: str, **_: Any):  # type: ignore[no-untyped-def]
        if self._stack:
            trace_id = self._stack[-1].trace_id
        else:
            trace_id = self._next_trace_id
            self._next_trace_id += 1

        span = _FakeSpan(name=name, trace_id=trace_id)
        self.spans.append(span)
        self._stack.append(span)
        try:
            yield span
        finally:
            self._stack.pop()


class _BrokenTracer:
    def start_span(self, name: str, **_: Any) -> Any:
        raise RuntimeError(f"broken tracer for {name}")

    @contextmanager
    def start_as_current_span(self, name: str, **_: Any):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"broken tracer for {name}")
        yield


class _FakeLocalParticipant:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    async def publish_data(
        self,
        *,
        payload: bytes,
        topic: str,
        reliable: bool,
    ) -> None:
        self.published.append(
            {
                "payload": payload,
                "topic": topic,
                "reliable": reliable,
            }
        )


class _FakeRoom:
    def __init__(self) -> None:
        self.name = "voice-test"
        self.local_participant = _FakeLocalParticipant()


def _decode_payloads(room: _FakeRoom) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for item in room.local_participant.published:
        if item["topic"] != "metrics":
            continue
        decoded.append(json.loads(item["payload"].decode("utf-8")))
    return decoded


def _make_stt_metrics(request_id: str) -> metrics.STTMetrics:
    return metrics.STTMetrics(
        label="stt",
        request_id=request_id,
        timestamp=0.0,
        duration=0.2,
        audio_duration=0.25,
        streamed=True,
    )


def _make_llm_metrics(
    speech_id: str,
    *,
    duration: float = 0.6,
    ttft: float = 0.1,
) -> metrics.LLMMetrics:
    return metrics.LLMMetrics(
        label="llm",
        request_id=f"req-{speech_id}",
        timestamp=0.0,
        duration=duration,
        ttft=ttft,
        cancelled=False,
        completion_tokens=24,
        prompt_tokens=12,
        prompt_cached_tokens=0,
        total_tokens=36,
        tokens_per_second=40.0,
        speech_id=speech_id,
    )


def _make_tts_metrics(
    speech_id: str,
    *,
    ttfb: float = 0.15,
    duration: float = 0.5,
    audio_duration: float = 1.3,
    metadata: Any = None,
) -> metrics.TTSMetrics:
    return metrics.TTSMetrics(
        label="tts",
        request_id=f"req-{speech_id}",
        timestamp=0.0,
        ttfb=ttfb,
        duration=duration,
        audio_duration=audio_duration,
        cancelled=False,
        characters_count=42,
        streamed=True,
        speech_id=speech_id,
        metadata=metadata,
    )


def _make_eou_metrics(
    speech_id: str,
    delay: float = 1.2,
    transcription_delay: float = 0.0,
) -> metrics.EOUMetrics:
    return metrics.EOUMetrics(
        timestamp=0.0,
        end_of_utterance_delay=delay,
        transcription_delay=transcription_delay,
        on_user_turn_completed_delay=0.0,
        speech_id=speech_id,
    )


def test_live_update_includes_eou_latencies_without_llm_or_tts() -> None:
    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=False,
    )

    async def _run() -> None:
        await collector.on_metrics_collected(
            _make_eou_metrics(
                "speech-eou-only",
                delay=0.9,
                transcription_delay=0.2,
            )
        )

    asyncio.run(_run())

    payloads = _decode_payloads(room)
    live_updates = [
        payload
        for payload in payloads
        if payload.get("type") == "metrics_live_update"
        and payload.get("stage") == "eou"
        and payload.get("speech_id") == "speech-eou-only"
    ]
    assert live_updates

    eou_update = live_updates[-1]
    assert eou_update["latencies"]["eou_delay"] == pytest.approx(0.9)
    assert eou_update["latencies"]["stt_finalization_delay"] == pytest.approx(0.2)
    assert eou_update["latencies"]["llm_ttft"] == 0.0
    assert eou_update["latencies"]["tts_ttfb"] == 0.0


@dataclass
class _FakeChatItem:
    role: str
    content: list[Any]
    created_at: float | None = None


class _FakeTextMethodPart:
    def __init__(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text


class _FakeSpeechHandle:
    def __init__(self, chat_items: list[Any], speech_id: str = "speech-fake") -> None:
        self.id = speech_id
        self.chat_items = chat_items
        self._callbacks: list[Any] = []
        self._item_added_callbacks: list[Any] = []

    def add_done_callback(self, callback: Any) -> None:
        self._callbacks.append(callback)

    def _add_item_added_callback(self, callback: Any) -> None:
        self._item_added_callbacks.append(callback)

    def _remove_item_added_callback(self, callback: Any) -> None:
        self._item_added_callbacks = [
            registered for registered in self._item_added_callbacks if registered is not callback
        ]

    def add_chat_item(self, item: Any) -> None:
        self.chat_items.append(item)
        for callback in list(self._item_added_callbacks):
            callback(item)

    def trigger_done(self) -> None:
        for callback in self._callbacks:
            callback(self)


class _FakeSpeechHandleWithoutItemAddedHook:
    def __init__(self, chat_items: list[Any], speech_id: str = "speech-fake") -> None:
        self.id = speech_id
        self.chat_items = chat_items
        self._callbacks: list[Any] = []

    def add_done_callback(self, callback: Any) -> None:
        self._callbacks.append(callback)

    def trigger_done(self) -> None:
        for callback in self._callbacks:
            callback(self)


@dataclass
class _FakeFunctionCall:
    name: str
    call_id: str
    arguments: str
    created_at: float


@dataclass
class _FakeFunctionCallOutput:
    output: str
    is_error: bool
    created_at: float


def test_turn_trace_has_required_metadata_and_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)
    monkeypatch.setattr(
        metrics_collector_module.settings.langfuse,
        "LANGFUSE_PUBLIC_TRACES",
        False,
    )

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-initial",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-abc",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hello there", is_final=True)
        await collector.on_metrics_collected(_make_stt_metrics("stt-1"))
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-1", delay=1.1, transcription_delay=0.25)
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-1"))
        await collector.on_conversation_item_added(role="assistant", content="hi, how can I help?")
        await collector.on_metrics_collected(_make_tts_metrics("speech-1"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    span_names = [span.name for span in fake_tracer.spans]
    assert span_names == [
        "turn",
        "user_input",
        "VADMetrics",
        "STTMetrics",
        "EOUMetrics",
        "agent_response_phase_1",
        "LLMMetrics",
        "TTSMetrics",
        "metrics_summary",
        "perceived_latency_first_audio",
    ]

    root = fake_tracer.spans[0]
    user_input_span = next(span for span in fake_tracer.spans if span.name == "user_input")
    vad_span = next(span for span in fake_tracer.spans if span.name == "VADMetrics")
    stt_span = next(span for span in fake_tracer.spans if span.name == "STTMetrics")
    eou_span = next(span for span in fake_tracer.spans if span.name == "EOUMetrics")
    phase_span = next(
        span for span in fake_tracer.spans if span.name == "agent_response_phase_1"
    )
    llm_span = next(span for span in fake_tracer.spans if span.name == "LLMMetrics")
    tts_span = next(span for span in fake_tracer.spans if span.name == "TTSMetrics")
    summary_span = next(span for span in fake_tracer.spans if span.name == "metrics_summary")
    perceived_first_span = next(
        span
        for span in fake_tracer.spans
        if span.name == "perceived_latency_first_audio"
    )
    assert root.attributes["session_id"] == "session-abc"
    assert root.attributes["room_id"] == "RM123"
    assert root.attributes["participant_id"] == "web-123"
    assert root.attributes["turn_id"]
    assert root.attributes["langfuse.trace.output"] == "hi, how can I help?"
    assert root.attributes["langfuse.trace.public"] is False
    assert root.attributes["langfuse.trace.metadata.finalization_reason"] == "complete"
    assert root.attributes["langfuse.trace.metadata.assistant_text_source"] == "conversation_item"
    assert root.attributes["latency_ms.eou_delay"] == pytest.approx(1100.0)
    assert root.attributes["latency_ms.stt_finalization"] == pytest.approx(250.0)
    assert root.attributes["latency_ms.stt_total"] == pytest.approx(1350.0)
    assert root.attributes["latency_ms.llm_ttft"] > 0
    assert root.attributes["latency_ms.llm_total"] > 0
    assert root.attributes["latency_ms.tts_ttfb"] > 0
    assert root.attributes["latency_ms.perceived_first_audio"] == pytest.approx(1350.0)
    assert root.attributes["latency_ms.conversational"] == pytest.approx(1350.0)
    assert root.attributes["latency_ms.speech_end_to_assistant_speech_start"] == pytest.approx(1350.0)

    assert user_input_span.attributes["user_transcript"] == "hello there"
    assert vad_span.attributes["eou_delay_ms"] == pytest.approx(1100.0)

    assert eou_span.attributes["duration_ms"] == pytest.approx(1100.0)
    assert eou_span.attributes["end_of_utterance_delay"] == pytest.approx(1.1)
    assert eou_span.attributes["transcription_delay"] == pytest.approx(0.25)
    assert eou_span.attributes["on_user_turn_completed_delay"] == pytest.approx(0.0)
    assert eou_span.attributes["speech_id"] == "speech-1"

    assert stt_span.attributes["user_transcript"] == "hello there"
    assert stt_span.attributes["stt_status"] == "measured"
    assert stt_span.attributes["duration_ms"] == pytest.approx(200.0)
    assert stt_span.attributes["request_id"] == "stt-1"
    assert stt_span.attributes["streamed"] is True
    assert stt_span.attributes["stt_finalization_ms"] == pytest.approx(250.0)
    assert stt_span.attributes["stt_total_latency_ms"] == pytest.approx(1350.0)

    assert llm_span.attributes["prompt_text"] == "hello there"
    assert llm_span.attributes["response_text"] == "hi, how can I help?"
    assert llm_span.attributes["ttft_ms"] > 0
    assert llm_span.attributes["llm_total_latency_ms"] > 0
    assert llm_span.attributes["total_duration_ms"] == pytest.approx(
        llm_span.attributes["llm_total_latency_ms"]
    )
    assert llm_span.attributes["input"] == "hello there"
    assert llm_span.attributes["output"] == "hi, how can I help?"
    assert llm_span.attributes["duration_ms"] == pytest.approx(
        llm_span.attributes["ttft_ms"]
    )
    assert llm_span.attributes["prompt_tokens"] == 12
    assert llm_span.attributes["completion_tokens"] == 24

    assert tts_span.attributes["assistant_text"] == "hi, how can I help?"
    assert tts_span.attributes["ttfb_ms"] > 0
    assert tts_span.attributes["tts_total_latency_ms"] > 0
    assert tts_span.attributes["total_duration_ms"] == pytest.approx(
        tts_span.attributes["tts_total_latency_ms"]
    )
    assert tts_span.attributes["input"] == "hi, how can I help?"
    assert tts_span.attributes["output"] == "hi, how can I help?"
    assert tts_span.attributes["duration_ms"] == pytest.approx(
        tts_span.attributes["ttfb_ms"]
    )
    assert tts_span.attributes["characters_count"] == 42
    assert tts_span.attributes["streamed"] is True

    assert phase_span.attributes["phase.index"] == 1
    assert phase_span.attributes["phase.kind"] == "single"
    assert summary_span.attributes["duration_ms"] >= perceived_first_span.attributes["duration_ms"]

    assert perceived_first_span.attributes["duration_ms"] == pytest.approx(1350.0)
    assert (
        perceived_first_span.attributes["speech_end_to_assistant_speech_start_ms"]
        == pytest.approx(1350.0)
    )
    assert perceived_first_span.attributes["eou_delay_ms"] == pytest.approx(1100.0)
    assert perceived_first_span.attributes["llm_ttft_ms"] > 0
    assert perceived_first_span.attributes["tts_ttfb_ms"] > 0
    assert all(span.end_count == 1 for span in fake_tracer.spans)

    payloads = _decode_payloads(room)
    conversation_turns = [
        payload for payload in payloads if payload.get("type") == "conversation_turn"
    ]
    agent_turn = next(
        payload
        for payload in conversation_turns
        if payload.get("role") == "agent"
    )
    assert agent_turn["latencies"]["total_latency"] == pytest.approx(
        root.attributes["latency_ms.conversational"] / 1000.0
    )
    assert agent_turn["metrics"]["llm"]["ttft"] == pytest.approx(
        root.attributes["latency_ms.llm_ttft"] / 1000.0
    )
    assert agent_turn["metrics"]["tts"]["ttfb"] == pytest.approx(
        root.attributes["latency_ms.tts_ttfb"] / 1000.0
    )

    trace_updates = [payload for payload in payloads if payload.get("type") == "trace_update"]
    assert len(trace_updates) == 1
    assert trace_updates[0]["session_id"] == "session-abc"
    assert trace_updates[0]["trace_id"]


def test_turn_trace_marks_public_when_langfuse_public_traces_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)
    monkeypatch.setattr(
        metrics_collector_module.settings.langfuse,
        "LANGFUSE_PUBLIC_TRACES",
        True,
    )

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-public",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(_make_stt_metrics("stt-public"))
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-public", delay=0.8, transcription_delay=0.1)
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-public"))
        await collector.on_conversation_item_added(role="assistant", content="hi")
        await collector.on_metrics_collected(_make_tts_metrics("speech-public"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    root = fake_tracer.spans[0]
    assert root.attributes["langfuse.trace.public"] is True


def test_turn_trace_includes_tool_spans_between_llm_and_tts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-tools",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("find papers", is_final=True)
        await collector.on_metrics_collected(_make_stt_metrics("stt-tools"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-tools"))
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-1",
                    arguments='{"query":"transformers"}',
                    created_at=100.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[{"title":"Attention Is All You Need"}]}',
                    is_error=False,
                    created_at=100.4,
                )
            ],
            created_at=100.4,
        )
        await collector.on_conversation_item_added(role="assistant", content="I found relevant papers.")
        await collector.on_metrics_collected(_make_tts_metrics("speech-tools"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    span_names = [span.name for span in fake_tracer.spans]
    assert span_names == [
        "turn",
        "user_input",
        "VADMetrics",
        "STTMetrics",
        "EOUMetrics",
        "agent_response_phase_1",
        "LLMMetrics",
        "tool_execution_1",
        "ToolCall",
        "agent_response_phase_2",
        "TTSMetrics",
        "metrics_summary",
        "perceived_latency_second_audio",
    ]

    phase_pre = next(
        span for span in fake_tracer.spans if span.name == "agent_response_phase_1"
    )
    phase_post = next(
        span for span in fake_tracer.spans if span.name == "agent_response_phase_2"
    )
    assert phase_pre.attributes["phase.kind"] == "pre-tool"
    assert phase_post.attributes["phase.kind"] == "post-tool"

    tool_span = next(span for span in fake_tracer.spans if span.name == "ToolCall")
    assert tool_span.attributes["tool.name"] == "paper_search"
    assert tool_span.attributes["tool.call_id"] == "call-1"
    assert tool_span.attributes["tool.is_error"] is False
    assert tool_span.attributes["duration_ms"] == pytest.approx(400.0)
    assert tool_span.attributes["input"] == '{"query":"transformers"}'
    assert tool_span.attributes["output"] == '{"results":[{"title":"Attention Is All You Need"}]}'
    assert tool_span.attributes["langfuse.observation.input"] == '{"query":"transformers"}'
    assert tool_span.attributes["langfuse.observation.output"] == '{"results":[{"title":"Attention Is All You Need"}]}'

    root = fake_tracer.spans[0]
    assert root.attributes["tool.call_count"] == 1
    assert root.attributes["tool.error_count"] == 0
    assert root.attributes["tool.execution_count"] == 1
    assert root.attributes["latency_ms.tool_calls_total"] == pytest.approx(400.0)
    assert root.attributes["latency_ms.perceived_second_audio"] >= 0


def test_turn_pipeline_summary_simple_turn_without_langfuse() -> None:
    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=False,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-summary-simple",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(_make_stt_metrics("stt-summary-simple"))
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-summary-simple", delay=0.8, transcription_delay=0.1)
        )
        await collector.on_metrics_collected(
            _make_llm_metrics("speech-summary-simple", ttft=0.12)
        )
        await collector.on_conversation_item_added(
            role="assistant",
            content="Hello! How can I help?",
        )
        await collector.on_metrics_collected(
            _make_tts_metrics("speech-summary-simple", ttfb=0.14, duration=0.4, audio_duration=0.8)
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    payloads = _decode_payloads(room)
    summaries = [
        payload for payload in payloads if payload.get("type") == "turn_pipeline_summary"
    ]
    assert len(summaries) == 1
    summary = summaries[0]

    assert summary["has_tools"] is False
    assert summary["tool_phase"] is None
    assert summary["post_tool_phases"] == []
    assert [phase["id"] for phase in summary["phases"]] == [1, 2, 3]
    assert summary["first_audio_latency_seconds"] is not None
    assert summary["second_audio_latency_seconds"] is None
    assert summary["total_turn_duration_seconds"] >= summary["first_audio_latency_seconds"]


def test_turn_pipeline_summary_tool_turn_contains_breakdown() -> None:
    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=False,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-summary-tools",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("search and summarize", is_final=True)
        await collector.on_metrics_collected(_make_stt_metrics("stt-summary-tools"))
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-summary-tools", delay=0.7, transcription_delay=0.05)
        )
        await collector.on_metrics_collected(
            _make_llm_metrics("speech-summary-tools", ttft=0.09)
        )
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="search_web",
                    call_id="call-summary-1",
                    arguments='{"query":"transformers"}',
                    created_at=20.0,
                ),
                _FakeFunctionCall(
                    name="get_weather",
                    call_id="call-summary-2",
                    arguments='{"city":"Bogota"}',
                    created_at=20.3,
                ),
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[{"title":"Attention Is All You Need"}]}',
                    is_error=False,
                    created_at=20.2,
                ),
                _FakeFunctionCallOutput(
                    output='{"temperature_c":18}',
                    is_error=False,
                    created_at=20.55,
                ),
            ],
            created_at=20.55,
        )
        await collector.on_conversation_item_added(
            role="assistant",
            content="I found relevant results and weather context.",
        )
        await collector.on_metrics_collected(
            _make_tts_metrics("speech-summary-tools", ttfb=0.18, duration=0.5, audio_duration=0.9)
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    payloads = _decode_payloads(room)
    summaries = [
        payload for payload in payloads if payload.get("type") == "turn_pipeline_summary"
    ]
    assert summaries
    partial_summaries = [payload for payload in summaries if payload.get("partial") is True]
    final_summaries = [payload for payload in summaries if payload.get("partial") is False]
    assert partial_summaries, "expected progressive tool summaries before final summary"
    assert len(final_summaries) == 1
    summary = final_summaries[0]

    assert summary["has_tools"] is True
    assert [phase["id"] for phase in summary["phases"]] == [1, 2, 3]
    assert [phase["id"] for phase in summary["post_tool_phases"]] == [5, 6]
    assert summary["tool_phase"]["execution_count"] == 1
    assert len(summary["tool_phase"]["tools"]) == 2
    assert summary["tool_phase"]["tools"][0]["name"] == "search_web"
    assert summary["tool_phase"]["tools"][1]["name"] == "get_weather"
    assert summary["second_audio_latency_seconds"] is not None
    assert summary["total_turn_duration_seconds"] >= summary["first_audio_latency_seconds"]


def test_turn_pipeline_summary_publishes_partial_at_tool_step_started() -> None:
    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=False,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-summary-step-start",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("use tools", is_final=True)
        await collector.on_metrics_collected(_make_stt_metrics("stt-summary-step-start"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-summary-step-start", ttft=0.11))
        await collector.on_tool_step_started()

    asyncio.run(_run())

    payloads = _decode_payloads(room)
    summaries = [
        payload for payload in payloads if payload.get("type") == "turn_pipeline_summary"
    ]
    assert summaries
    partial = summaries[-1]
    assert partial["partial"] is True
    assert partial["has_tools"] is True
    assert partial["tool_phase"] is not None
    assert partial["tool_phase"]["tools"] == []
    assert [phase["id"] for phase in partial["post_tool_phases"]] == [5, 6]


def test_turn_trace_supports_multiple_tool_execution_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-tools-multi-round",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("find and compare papers", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-tools-multi", ttft=0.08))
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-round-1",
                    arguments='{"query":"transformers"}',
                    created_at=10.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[{"title":"Attention Is All You Need"}]}',
                    is_error=False,
                    created_at=10.2,
                )
            ],
            created_at=10.2,
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-tools-multi", ttft=0.09))
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_compare",
                    call_id="call-round-2",
                    arguments='{"left":"A","right":"B"}',
                    created_at=10.3,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"winner":"A"}',
                    is_error=False,
                    created_at=10.5,
                )
            ],
            created_at=10.5,
        )
        await collector.on_conversation_item_added(
            role="assistant",
            content="I compared them. Paper A is more relevant.",
        )
        await collector.on_metrics_collected(
            _make_tts_metrics("speech-tools-multi", ttfb=0.12, duration=0.4, audio_duration=0.4)
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    span_names = [span.name for span in fake_tracer.spans]
    assert span_names.count("LLMMetrics") == 2
    assert span_names.count("ToolCall") == 2
    assert "tool_execution_1" in span_names
    assert "tool_execution_2" in span_names
    assert "agent_response_phase_3" in span_names
    assert "perceived_latency_second_audio" in span_names

    root = fake_tracer.spans[0]
    assert root.attributes["tool.call_count"] == 2
    assert root.attributes["tool.execution_count"] == 2
    assert root.attributes["langfuse.trace.output"] == "I compared them. Paper A is more relevant."
    assert root.attributes["latency_ms.perceived_second_audio"] >= 0


def test_tool_span_keeps_full_input_output_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    huge_args = "x" * 12000
    huge_output = "y" * 13000

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-tools-full-payload",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("run tool", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-tools-full"))
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-full",
                    arguments=huge_args,
                    created_at=10.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output=huge_output,
                    is_error=False,
                    created_at=10.2,
                )
            ],
            created_at=10.2,
        )
        await collector.on_conversation_item_added(role="assistant", content="done")
        await collector.on_metrics_collected(_make_tts_metrics("speech-tools-full"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    tool_span = next(span for span in fake_tracer.spans if span.name == "ToolCall")
    assert tool_span.attributes["input"] == huge_args
    assert tool_span.attributes["output"] == huge_output


def test_tool_error_span_marks_error_and_root_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-tools-error",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("search this", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-tools-error"))
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-error",
                    arguments='{"query":""}',
                    created_at=50.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output="MCP error -32602: invalid query",
                    is_error=True,
                    created_at=50.1,
                )
            ],
            created_at=50.1,
        )
        await collector.on_conversation_item_added(role="assistant", content="Tool failed.")
        await collector.on_metrics_collected(_make_tts_metrics("speech-tools-error"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    tool_span = next(span for span in fake_tracer.spans if span.name == "ToolCall")
    assert tool_span.attributes["tool.is_error"] is True

    root = fake_tracer.spans[0]
    assert root.attributes["tool.call_count"] == 1
    assert root.attributes["tool.error_count"] == 1


def test_tool_span_survives_when_tts_precedes_function_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-tools-out-of-order",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("find papers", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-tools-out-of-order"))
        await collector.on_tool_step_started()

        # Pre-tool lead-in speech can arrive before tool execution callbacks.
        await collector.on_conversation_item_added(role="assistant", content="Let me check that.")
        await collector.on_metrics_collected(
            _make_tts_metrics("speech-tools-out-of-order", ttfb=0.08, duration=0.2, audio_duration=0.2)
        )

        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-late",
                    arguments='{"query":"transformers"}',
                    created_at=100.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[{"title":"Attention Is All You Need"}]}',
                    is_error=False,
                    created_at=100.4,
                )
            ],
            created_at=100.4,
        )
        await collector.on_conversation_item_added(role="assistant", content="I found relevant papers.")
        await collector.on_metrics_collected(
            _make_tts_metrics("speech-tools-out-of-order", ttfb=0.15, duration=0.4, audio_duration=0.4)
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    span_names = [span.name for span in fake_tracer.spans]
    assert "ToolCall" in span_names
    tool_span = next(span for span in fake_tracer.spans if span.name == "ToolCall")
    assert tool_span.attributes["tool.call_id"] == "call-late"

    root = fake_tracer.spans[0]
    assert (
        root.attributes["langfuse.trace.output"]
        == "Let me check that.\nI found relevant papers."
    )
    assert root.attributes["tool.phase_announced"] is True
    assert root.attributes["tool.post_response_missing"] is False


def test_trace_waits_for_post_tool_response_before_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-tools-wait-post-response",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("use tools", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-tools-wait-post"))
        await collector.on_tool_step_started()
        await collector.on_conversation_item_added(role="assistant", content="One sec, checking.")
        await collector.on_metrics_collected(_make_tts_metrics("speech-tools-wait-post"))
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-wait",
                    arguments='{"query":"agents"}',
                    created_at=200.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[{"title":"Toolformer"}]}',
                    is_error=False,
                    created_at=200.2,
                )
            ],
            created_at=200.2,
        )
        await collector.wait_for_pending_trace_tasks()
        assert not fake_tracer.spans

        await collector.on_conversation_item_added(role="assistant", content="Here are the top results.")
        await collector.on_metrics_collected(_make_tts_metrics("speech-tools-wait-post"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    span_names = [span.name for span in fake_tracer.spans]
    assert "ToolCall" in span_names
    root = fake_tracer.spans[0]
    assert (
        root.attributes["langfuse.trace.output"]
        == "One sec, checking.\nHere are the top results."
    )


def test_delayed_pre_tool_assistant_event_uses_timestamps_and_keeps_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    base_time = time.time() - 2.0
    pre_tool_text_created_at = base_time + 0.1
    tool_created_at = base_time + 0.2
    final_text_created_at = base_time + 0.4
    speech_id = "speech-delayed-pre-tool-assistant"

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-delayed-pre-tool-assistant",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("Graphic design", is_final=True)
        await collector.on_metrics_collected(
            _make_llm_metrics(speech_id, ttft=0.08, duration=0.35)
        )
        await collector.on_tool_step_started()

        # Pre-tool lead-in TTS can be observed before conversation item events.
        await collector.on_metrics_collected(
            _make_tts_metrics(speech_id, ttfb=0.06, duration=0.2, audio_duration=0.2)
        )

        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-delayed-pre-tool-assistant",
                    arguments='{"query":"graphic design"}',
                    created_at=tool_created_at - 0.05,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[{"title":"Thinking with Type"}]}',
                    is_error=False,
                    created_at=tool_created_at,
                )
            ],
            created_at=tool_created_at,
        )

        # Delayed pre-tool lead-in arrives after tool execution callback.
        await collector.on_conversation_item_added(
            role="assistant",
            content="Checking that for you.",
            event_created_at=tool_created_at + 1.0,
            item_created_at=pre_tool_text_created_at,
        )

        await collector.on_metrics_collected(
            _make_llm_metrics(speech_id, ttft=0.12, duration=0.45)
        )
        await collector.on_metrics_collected(
            _make_tts_metrics(speech_id, ttfb=0.11, duration=0.6, audio_duration=0.6)
        )
        await collector.wait_for_pending_trace_tasks()
        assert not fake_tracer.spans

        await collector.on_conversation_item_added(
            role="assistant",
            content="Graphic design is the craft of visual communication.",
            event_created_at=final_text_created_at,
            item_created_at=final_text_created_at,
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    root = next(span for span in fake_tracer.spans if span.name == "turn")
    assert root.attributes["tool.post_response_missing"] is False
    assert (
        root.attributes["langfuse.trace.output"]
        == "Checking that for you.\nGraphic design is the craft of visual communication."
    )

    tts_phase_2 = next(
        span
        for span in fake_tracer.spans
        if span.name == "TTSMetrics" and span.attributes.get("phase_index") == 2
    )
    assert (
        tts_phase_2.attributes["assistant_text"]
        == "Graphic design is the craft of visual communication."
    )


def test_timeout_finalizes_tool_turn_with_missing_post_tool_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01
    collector._tracer._trace_legacy_finalize_timeout_sec = 0.03
    collector._tracer._trace_legacy_finalize_timeout_sec = 0.03
    collector._trace_post_tool_response_timeout_sec = 0.01

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-tools-timeout",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("run tool", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-tools-timeout"))
        await collector.on_tool_step_started()
        await collector.on_conversation_item_added(role="assistant", content="Checking now.")
        await collector.on_metrics_collected(_make_tts_metrics("speech-tools-timeout"))
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-timeout",
                    arguments='{"query":"timeout"}',
                    created_at=300.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[{"title":"Delayed"}]}',
                    is_error=False,
                    created_at=300.5,
                )
            ],
            created_at=300.5,
        )
        await asyncio.sleep(0.05)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    span_names = [span.name for span in fake_tracer.spans]
    assert "ToolCall" in span_names
    root = fake_tracer.spans[0]
    assert root.attributes["tool.phase_announced"] is True
    assert root.attributes["tool.post_response_missing"] is True
    assert root.attributes["langfuse.trace.output"] == "[assistant text unavailable]"


def test_timeout_tool_error_turn_uses_fallback_error_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01
    collector._trace_post_tool_response_timeout_sec = 0.01

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-tools-timeout-error-summary",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("run tool", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-tools-timeout-error-summary"))
        await collector.on_tool_step_started()
        await collector.on_conversation_item_added(role="assistant", content="Checking now.")
        await collector.on_metrics_collected(_make_tts_metrics("speech-tools-timeout-error-summary"))
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-timeout-error-summary",
                    arguments='{"query":"timeout"}',
                    created_at=301.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output="MCP error -32602: invalid query",
                    is_error=True,
                    created_at=301.2,
                )
            ],
            created_at=301.2,
        )
        await asyncio.sleep(0.05)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    root = fake_tracer.spans[0]
    assert root.attributes["tool.post_response_missing"] is True
    assert root.attributes["tool.error_count"] == 1
    assert (
        root.attributes["langfuse.trace.output"]
        == "I couldn't complete that tool request. Please rephrase the query."
    )


def test_post_tool_assistant_text_without_post_tool_tts_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01
    collector._trace_post_tool_response_timeout_sec = 0.01

    async def _run() -> None:
        speech_id = "speech-post-tool-no-tts"
        await collector.on_session_metadata(
            session_id="session-post-tool-no-tts",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("find one paper", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics(speech_id, ttft=0.08))
        await collector.on_tool_step_started()
        await collector.on_conversation_item_added(role="assistant", content="I'll look that up.")
        await collector.on_metrics_collected(
            _make_tts_metrics(speech_id, ttfb=0.07, duration=0.2, audio_duration=0.2)
        )
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-post-tool-no-tts",
                    arguments='{"query":"medicine"}',
                    created_at=600.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[{"title":"GAIA"}]}',
                    is_error=False,
                    created_at=600.3,
                )
            ],
            created_at=600.3,
        )
        await collector.on_metrics_collected(_make_llm_metrics(speech_id, ttft=0.11))
        await collector.on_conversation_item_added(
            role="assistant",
            content="The key paper is GAIA.",
        )
        await asyncio.sleep(0.05)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    root = fake_tracer.spans[0]
    assert (
        root.attributes["langfuse.trace.output"]
        == "I'll look that up.\nThe key paper is GAIA."
    )


def test_post_tool_timeout_prevents_early_finalize_of_pre_tool_leadin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01
    collector._trace_post_tool_response_timeout_sec = 0.08

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-post-tool-timeout-window",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("find me the best paper", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-post-tool-timeout-window"))
        await collector.on_tool_step_started()
        await collector.on_conversation_item_added(role="assistant", content="I'll look that up.")
        await collector.on_metrics_collected(_make_tts_metrics("speech-post-tool-timeout-window"))
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-post-tool-timeout-window",
                    arguments='{"query":"mps cubic phases"}',
                    created_at=400.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[{"title":"A key paper"}]}',
                    is_error=False,
                    created_at=400.2,
                )
            ],
            created_at=400.2,
        )

        # The base finalize timeout has elapsed, but post-tool timeout should keep the turn pending.
        await asyncio.sleep(0.03)
        await collector.wait_for_pending_trace_tasks()
        assert not fake_tracer.spans

        await collector.on_conversation_item_added(
            role="assistant",
            content="The most cited paper is Attention Is All You Need.",
        )
        await collector.on_metrics_collected(
            _make_tts_metrics("speech-post-tool-timeout-window")
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    root = fake_tracer.spans[0]
    assert (
        root.attributes["langfuse.trace.output"]
        == "I'll look that up.\nThe most cited paper is Attention Is All You Need."
    )
    assert root.attributes["tool.post_response_missing"] is False


def test_tool_turn_phase_text_order_and_root_output_concatenation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-phase-order",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("find one key medicine paper", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-phase-order", ttft=0.08))
        await collector.on_tool_step_started()
        await collector.on_conversation_item_added(role="assistant", content="I'll look that up.")
        await collector.on_metrics_collected(
            _make_tts_metrics("speech-phase-order", ttfb=0.07, duration=0.2, audio_duration=0.2)
        )

        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-phase-order",
                    arguments='{"query":"medicine"}',
                    created_at=500.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[{"title":"GAIA: a benchmark for General AI Assistants"}]}',
                    is_error=False,
                    created_at=500.4,
                )
            ],
            created_at=500.4,
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-phase-order", ttft=0.12))
        await collector.on_conversation_item_added(
            role="assistant",
            content='The key paper is "GAIA: a benchmark for General AI Assistants".',
        )
        await collector.on_metrics_collected(
            _make_tts_metrics("speech-phase-order", ttfb=0.11, duration=0.35, audio_duration=0.35)
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    phase_spans = [
        span for span in fake_tracer.spans if span.name.startswith("agent_response_phase_")
    ]
    assert len(phase_spans) == 2
    phase1 = next(span for span in phase_spans if span.attributes["phase.index"] == 1)
    phase2 = next(span for span in phase_spans if span.attributes["phase.index"] == 2)
    assert phase1.attributes["phase.kind"] == "pre-tool"
    assert phase2.attributes["phase.kind"] == "post-tool"

    llm_spans = [span for span in fake_tracer.spans if span.name == "LLMMetrics"]
    assert len(llm_spans) == 2
    llm_phase_1 = next(span for span in llm_spans if span.attributes["phase_index"] == 1)
    llm_phase_2 = next(span for span in llm_spans if span.attributes["phase_index"] == 2)
    assert llm_phase_1.attributes["response_text"] == "I'll look that up."
    assert (
        llm_phase_2.attributes["response_text"]
        == 'The key paper is "GAIA: a benchmark for General AI Assistants".'
    )

    tts_spans = [span for span in fake_tracer.spans if span.name == "TTSMetrics"]
    assert len(tts_spans) == 2
    tts_phase_1 = next(span for span in tts_spans if span.attributes["phase_index"] == 1)
    tts_phase_2 = next(span for span in tts_spans if span.attributes["phase_index"] == 2)
    assert tts_phase_1.attributes["assistant_text"] == "I'll look that up."
    assert (
        tts_phase_2.attributes["assistant_text"]
        == 'The key paper is "GAIA: a benchmark for General AI Assistants".'
    )

    root = fake_tracer.spans[0]
    assert (
        root.attributes["langfuse.trace.output"]
        == "I'll look that up.\nThe key paper is \"GAIA: a benchmark for General AI Assistants\"."
    )


def test_tool_event_without_matching_turn_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-no-tool-turn",
            participant_id="web-123",
        )
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="paper_search",
                    call_id="call-orphan",
                    arguments='{"query":"ignored"}',
                    created_at=1.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[]}',
                    is_error=False,
                    created_at=1.1,
                )
            ],
            created_at=1.1,
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-no-tool-turn"))
        await collector.on_conversation_item_added(role="assistant", content="hello")
        await collector.on_metrics_collected(_make_tts_metrics("speech-no-tool-turn"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    span_names = [span.name for span in fake_tracer.spans]
    assert "ToolCall" not in span_names


def test_tracing_failure_does_not_break_metrics_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    monkeypatch.setattr(metrics_collector_module, "tracer", _BrokenTracer())

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-xyz",
            participant_id="web-xyz",
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(_make_stt_metrics("stt-2"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-2"))
        await collector.on_conversation_item_added(role="assistant", content="hello back")
        await collector.on_metrics_collected(_make_tts_metrics("speech-2"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    payloads = _decode_payloads(room)
    conversation_turns = [payload for payload in payloads if payload.get("type") == "conversation_turn"]
    assert conversation_turns, "metrics publishing should still work when tracing fails"


def test_creates_new_trace_for_each_finalized_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-two-turns",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("partial", is_final=False)
        await collector.on_user_input_transcribed("first turn", is_final=True)
        await collector.on_metrics_collected(_make_stt_metrics("stt-a"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-a"))
        await collector.on_conversation_item_added(role="assistant", content="first reply")
        await collector.on_metrics_collected(_make_tts_metrics("speech-a"))

        await collector.on_user_input_transcribed("second turn", is_final=True)
        await collector.on_metrics_collected(_make_stt_metrics("stt-b"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-b"))
        await collector.on_conversation_item_added(role="assistant", content="second reply")
        await collector.on_metrics_collected(_make_tts_metrics("speech-b"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 2
    assert turn_spans[0].trace_id != turn_spans[1].trace_id


def test_multiple_final_transcripts_are_merged_into_one_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-merged-finals",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("What", is_final=True)
        await collector.on_user_input_transcribed(
            "the difference between speech to text and speech recognition?",
            is_final=True,
        )
        await collector.on_metrics_collected(_make_stt_metrics("stt-merged"))
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-merged", delay=0.9, transcription_delay=0.2)
        )
        await collector.on_conversation_item_added(
            role="user",
            content="What the difference between speech to text and speech recognition?",
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-merged"))
        await collector.on_conversation_item_added(role="assistant", content="Speech to text writes words down.")
        await collector.on_metrics_collected(_make_tts_metrics("speech-merged"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    stt_span = next(span for span in fake_tracer.spans if span.name == "STTMetrics")
    assert (
        root.attributes["langfuse.trace.input"]
        == "What the difference between speech to text and speech recognition?"
    )
    assert (
        stt_span.attributes["user_transcript"]
        == "What the difference between speech to text and speech recognition?"
    )
    assert root.attributes["langfuse.trace.metadata.coalesced_turn_count"] == 0


def test_same_speech_final_transcripts_keep_merging_after_eou_until_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-same-speech-final-merge-after-eou",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("Hello there.", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-final-merge-after-eou", delay=0.9, transcription_delay=0.2)
        )
        await collector.on_user_input_transcribed("I'm missing context.", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-final-merge-after-eou"))
        await collector.on_conversation_item_added(
            role="assistant",
            content="Hi there.",
        )
        await collector.on_metrics_collected(_make_tts_metrics("speech-final-merge-after-eou"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.input"] == "Hello there. I'm missing context."
    assert root.attributes["langfuse.trace.output"] == "Hi there."
    assert root.attributes["langfuse.trace.metadata.assistant_text_missing"] is False


def test_user_conversation_item_after_eou_merges_instead_of_replacing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-user-item-merges-after-eou",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("Hello there.", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-user-item-after-eou", delay=0.8, transcription_delay=0.2)
        )
        await collector.on_conversation_item_added(
            role="user",
            content="I'm missing the rest.",
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-user-item-after-eou"))
        await collector.on_conversation_item_added(
            role="assistant",
            content="Thanks for clarifying.",
        )
        await collector.on_metrics_collected(_make_tts_metrics("speech-user-item-after-eou"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.input"] == "Hello there. I'm missing the rest."
    assert root.attributes["langfuse.trace.output"] == "Thanks for clarifying."


def test_same_speech_fragmented_input_with_late_speech_done_keeps_full_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    handle = _FakeSpeechHandle(chat_items=[], speech_id="speech-fragmented-late-done")

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-fragmented-late-done",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("Hello there.", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-fragmented-late-done", delay=0.8, transcription_delay=0.2)
        )
        await collector.on_user_input_transcribed("I'm missing context.", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-fragmented-late-done"))
        await collector.on_speech_created(handle)
        await collector.on_metrics_collected(_make_tts_metrics("speech-fragmented-late-done"))
        await collector.wait_for_pending_trace_tasks()

        turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
        assert not turn_spans

        handle.chat_items = [_FakeChatItem(role="assistant", content=["Hi there."])]
        handle.trigger_done()
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.input"] == "Hello there. I'm missing context."
    assert root.attributes["langfuse.trace.output"] == "Hi there."
    assert root.attributes["langfuse.trace.metadata.assistant_text_missing"] is False


def test_immediate_continuation_after_tts_does_not_mark_coalesced_turn_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)
    monkeypatch.setattr(
        metrics_collector_module.settings.langfuse,
        "LANGFUSE_CONTINUATION_COALESCE_WINDOW_MS",
        1500.0,
    )

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        base_time = time.time()
        await collector.on_session_metadata(
            session_id="session-coalesce",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("What", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-a", delay=0.7, transcription_delay=0.2)
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-a"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-a"))

        await collector.on_user_input_transcribed(
            "the difference between speech to text and speech recognition?",
            is_final=True,
        )
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-b", delay=0.7, transcription_delay=0.2)
        )
        await collector.on_conversation_item_added(
            role="user",
            content="What the difference between speech to text and speech recognition?",
            event_created_at=base_time + 0.2,
            item_created_at=base_time + 0.2,
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-b"))
        await collector.on_conversation_item_added(
            role="assistant",
            content="Speech to text writes words down.",
            event_created_at=base_time + 0.4,
            item_created_at=base_time + 0.4,
        )
        await collector.on_metrics_collected(_make_tts_metrics("speech-b"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert (
        root.attributes["langfuse.trace.input"]
        == "What the difference between speech to text and speech recognition?"
    )
    assert root.attributes["langfuse.trace.metadata.coalesced_turn_count"] == 0
    assert root.attributes["langfuse.trace.metadata.coalesced_fragment_count"] == 0
    assert root.attributes["langfuse.trace.metadata.coalesced_inputs"] == []


def test_visible_assistant_reply_prevents_continuation_coalescing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-no-coalesce-visible-reply",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("first turn", is_final=True)
        await collector.on_metrics_collected(_make_stt_metrics("stt-first-visible"))
        await collector.on_metrics_collected(_make_eou_metrics("speech-first-visible"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-first-visible"))
        await collector.on_conversation_item_added(role="assistant", content="first reply")
        await collector.on_metrics_collected(_make_tts_metrics("speech-first-visible"))

        await collector.on_user_input_transcribed("second turn", is_final=True)
        await collector.on_metrics_collected(_make_stt_metrics("stt-second-visible"))
        await collector.on_metrics_collected(_make_eou_metrics("speech-second-visible"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-second-visible"))
        await collector.on_conversation_item_added(role="assistant", content="second reply")
        await collector.on_metrics_collected(_make_tts_metrics("speech-second-visible"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 2
    assert turn_spans[0].attributes["langfuse.trace.metadata.coalesced_turn_count"] == 0
    assert turn_spans[1].attributes["langfuse.trace.metadata.coalesced_turn_count"] == 0


def test_tool_activity_prevents_continuation_coalescing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.05
    collector._trace_post_tool_response_timeout_sec = 0.05

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-no-coalesce-tools",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("run tool", is_final=True)
        await collector.on_metrics_collected(_make_eou_metrics("speech-tool-a"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-tool-a"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-tool-a"))
        await collector.on_tool_step_started()
        await collector.on_function_tools_executed(
            function_calls=[
                _FakeFunctionCall(
                    name="search_web",
                    call_id="tool-a",
                    arguments='{"q":"speech models"}',
                    created_at=1.0,
                )
            ],
            function_call_outputs=[
                _FakeFunctionCallOutput(
                    output='{"results":[]}',
                    is_error=False,
                    created_at=1.1,
                )
            ],
            created_at=1.1,
        )

        await collector.on_user_input_transcribed("follow-up turn", is_final=True)
        await collector.on_metrics_collected(_make_eou_metrics("speech-tool-b"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-tool-b"))
        await collector.on_conversation_item_added(role="assistant", content="second reply")
        await collector.on_metrics_collected(_make_tts_metrics("speech-tool-b"))
        await asyncio.sleep(0.08)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 2
    assert turn_spans[1].attributes["langfuse.trace.metadata.coalesced_turn_count"] == 0


def test_multiple_final_transcripts_share_one_llm_stall_watchdog() -> None:
    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=False,
    )

    async def _run() -> None:
        await collector.on_user_input_transcribed("Search for the most popular", is_final=True)
        await collector.on_user_input_transcribed("test to speech model.", is_final=True)
        assert len(collector._pending_user_utterances) == 1
        assert len(collector._llm_stall_tasks) == 1
        assert (
            collector._pending_user_utterances[0].transcript
            == "Search for the most popular test to speech model."
        )

        await collector.on_metrics_collected(_make_llm_metrics("speech-watchdog"))
        await asyncio.sleep(0)

    asyncio.run(_run())

    assert not collector._llm_stall_tasks


def test_false_turn_stall_reopens_user_utterance_before_response_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._llm_stall_timeout_sec = 0.01

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-false-turn-stall",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed(
            "What's the difference between speech to text",
            is_final=True,
        )
        await collector.on_metrics_collected(_make_eou_metrics("speech-false-turn-a"))
        await asyncio.sleep(0.02)

        assert len(collector._pending_user_utterances) == 1
        stalled_utterance = collector._pending_user_utterances[0]
        assert stalled_utterance.llm_stalled_before_response is True
        assert stalled_utterance.assistant_response_started is False

        await collector.on_user_input_transcribed(
            "and speech recognition?",
            is_final=True,
        )

        reopened_utterance = collector._pending_user_utterances[0]
        assert (
            reopened_utterance.transcript
            == "What's the difference between speech to text and speech recognition?"
        )
        assert reopened_utterance.speech_id is None
        assert reopened_utterance.llm_started is False
        assert reopened_utterance.llm_stalled_before_response is False

        await collector.on_metrics_collected(_make_eou_metrics("speech-false-turn-b"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-false-turn-b"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-false-turn-b"))
        collector.submit_streamed_assistant_text_delta(
            "speech-false-turn-b",
            "Speech-to-text converts spoken words into text, while speech recognition interprets the speech for meaning or action.",
            time.time(),
        )
        collector.submit_streamed_assistant_text_flush(
            "speech-false-turn-b",
            time.time(),
        )
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert (
        root.attributes["langfuse.trace.input"]
        == "What's the difference between speech to text and speech recognition?"
    )
    assert (
        root.attributes["langfuse.trace.output"]
        == "Speech-to-text converts spoken words into text, while speech recognition interprets the speech for meaning or action."
    )
    assert root.attributes["langfuse.trace.metadata.false_turn_recovered"] is True
    assert (
        root.attributes["langfuse.trace.metadata.merge_before_response_started"]
        is True
    )


def test_stale_pre_response_events_are_ignored_after_false_turn_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-false-turn-stale-events",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("Find papers about", is_final=True)
        await collector.on_metrics_collected(_make_eou_metrics("speech-stale-a"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-stale-a"))
        collector.submit_streamed_assistant_text_delta(
            "speech-stale-a",
            "Stale partial answer that should be dropped.",
            time.time(),
        )
        collector.submit_streamed_assistant_text_flush(
            "speech-stale-a",
            time.time(),
        )
        await asyncio.sleep(0)

        await collector.on_user_input_transcribed("voice agents.", is_final=True)

        await collector.on_metrics_collected(_make_eou_metrics("speech-stale-b"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-stale-b"))

        await collector.on_metrics_collected(_make_tts_metrics("speech-stale-a"))
        collector.submit_streamed_assistant_text_delta(
            "speech-stale-a",
            "Stale old response that must be ignored.",
            time.time(),
        )
        collector.submit_streamed_assistant_text_flush(
            "speech-stale-a",
            time.time(),
        )

        await collector.on_metrics_collected(_make_tts_metrics("speech-stale-b"))
        collector.submit_streamed_assistant_text_delta(
            "speech-stale-b",
            "Here are recent papers about voice agents.",
            time.time(),
        )
        collector.submit_streamed_assistant_text_flush(
            "speech-stale-b",
            time.time(),
        )
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.input"] == "Find papers about voice agents."
    assert (
        root.attributes["langfuse.trace.output"]
        == "Here are recent papers about voice agents."
    )
    assert (
        root.attributes["langfuse.trace.metadata.assistant_text_source"]
        == "text_output_stream"
    )
    assert root.attributes["langfuse.trace.metadata.false_turn_recovered"] is True


def test_premature_speaking_state_does_not_split_mergeable_user_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    final_input = "And what is a speech recognition?"
    final_output = "Speech recognition converts spoken language into written text."

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-premature-speaking-no-split",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("And what is a speech", is_final=True)
        await collector.on_metrics_collected(_make_eou_metrics("speech-speaking-a"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-speaking-a"))
        collector.submit_streamed_assistant_text_delta(
            "speech-speaking-a",
            "Stale partial answer that should be dropped.",
            time.time(),
        )
        collector.submit_streamed_assistant_text_flush(
            "speech-speaking-a",
            time.time(),
        )
        await asyncio.sleep(0)

        await collector.on_agent_state_changed(
            old_state="thinking",
            new_state="speaking",
        )
        assert len(collector._pending_user_utterances) == 1
        assert collector._pending_user_utterances[0].assistant_response_started is False

        await collector.on_user_input_transcribed("recognition?", is_final=True)
        reopened_utterance = collector._pending_user_utterances[0]
        assert reopened_utterance.transcript == final_input
        assert reopened_utterance.assistant_response_started is False

        await collector.on_metrics_collected(_make_eou_metrics("speech-speaking-b"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-speaking-b"))

        await collector.on_metrics_collected(_make_tts_metrics("speech-speaking-a"))
        collector.submit_streamed_assistant_text_delta(
            "speech-speaking-a",
            "Old response that must stay quarantined.",
            time.time(),
        )
        collector.submit_streamed_assistant_text_flush(
            "speech-speaking-a",
            time.time(),
        )

        await collector.on_metrics_collected(_make_tts_metrics("speech-speaking-b"))
        collector.submit_streamed_assistant_text_delta(
            "speech-speaking-b",
            final_output,
            time.time(),
        )
        collector.submit_streamed_assistant_text_flush(
            "speech-speaking-b",
            time.time(),
        )
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.input"] == final_input
    assert root.attributes["langfuse.trace.output"] == final_output
    assert root.attributes["langfuse.trace.metadata.false_turn_recovered"] is True
    assert (
        root.attributes["langfuse.trace.metadata.merge_before_response_started"]
        is True
    )


def test_stale_global_speaking_state_does_not_block_next_turn_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    second_input = "And what is a speech recognition?"
    second_output = "Speech recognition turns spoken words into structured text."

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-stale-global-speaking",
            participant_id="web-123",
        )

        await collector.on_user_input_transcribed("Hello there", is_final=True)
        await collector.on_metrics_collected(_make_eou_metrics("speech-prev"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-prev"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-prev"))
        collector.submit_streamed_assistant_text_delta(
            "speech-prev",
            "Hi, how can I help?",
            time.time(),
        )
        collector.submit_streamed_assistant_text_flush(
            "speech-prev",
            time.time(),
        )
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

        await collector.on_agent_state_changed(
            old_state="listening",
            new_state="speaking",
        )

        await collector.on_user_input_transcribed("And what is a speech", is_final=True)
        await collector.on_metrics_collected(_make_eou_metrics("speech-next-a"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-next-a"))
        collector.submit_streamed_assistant_text_delta(
            "speech-next-a",
            "Stale partial response.",
            time.time(),
        )
        collector.submit_streamed_assistant_text_flush(
            "speech-next-a",
            time.time(),
        )
        await asyncio.sleep(0)

        await collector.on_user_input_transcribed("recognition?", is_final=True)
        merged_utterance = collector._pending_user_utterances[-1]
        assert merged_utterance.transcript == second_input
        assert merged_utterance.assistant_response_started is False

        await collector.on_metrics_collected(_make_eou_metrics("speech-next-b"))
        await collector.on_metrics_collected(_make_llm_metrics("speech-next-b"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-next-b"))
        collector.submit_streamed_assistant_text_delta(
            "speech-next-b",
            second_output,
            time.time(),
        )
        collector.submit_streamed_assistant_text_flush(
            "speech-next-b",
            time.time(),
        )
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 2
    first = next(
        span
        for span in turn_spans
        if span.attributes["langfuse.trace.input"] == "Hello there"
    )
    second = next(
        span
        for span in turn_spans
        if span.attributes["langfuse.trace.input"] == second_input
    )
    assert first.attributes["langfuse.trace.output"] == "Hi, how can I help?"
    assert second.attributes["langfuse.trace.output"] == second_output
    assert second.attributes["langfuse.trace.metadata.false_turn_recovered"] is True


def test_continuation_coalescing_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)
    monkeypatch.setattr(
        metrics_collector_module.settings.langfuse,
        "LANGFUSE_CONTINUATION_COALESCE_WINDOW_MS",
        0.0,
    )

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01
    collector._tracer._trace_legacy_finalize_timeout_sec = 0.03

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-no-coalesce-disabled",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("What", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-disabled-a", delay=0.7, transcription_delay=0.2)
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-disabled-a"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-disabled-a"))
        await asyncio.sleep(0.03)

        await collector.on_user_input_transcribed(
            "the difference between speech to text and speech recognition?",
            is_final=True,
        )
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-disabled-b", delay=0.7, transcription_delay=0.2)
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-disabled-b"))
        await collector.on_conversation_item_added(role="assistant", content="Speech to text writes words down.")
        await collector.on_metrics_collected(_make_tts_metrics("speech-disabled-b"))
        await asyncio.sleep(0.05)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 2
    assert turn_spans[0].attributes["langfuse.trace.input"] == "What"
    assert (
        turn_spans[1].attributes["langfuse.trace.input"]
        == "the difference between speech to text and speech recognition?"
    )


def test_trace_emits_without_stt_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-no-stt-metrics",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("turn without stt metrics", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-no-stt"))
        await collector.on_conversation_item_added(role="assistant", content="reply without stt")
        await collector.on_metrics_collected(_make_tts_metrics("speech-no-stt"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    span_names = [span.name for span in fake_tracer.spans]
    assert span_names == [
        "turn",
        "user_input",
        "VADMetrics",
        "STTMetrics",
        "EOUMetrics",
        "agent_response_phase_1",
        "LLMMetrics",
        "TTSMetrics",
        "metrics_summary",
    ]
    stt_span = next(span for span in fake_tracer.spans if span.name == "STTMetrics")
    assert stt_span.attributes["user_transcript"] == "turn without stt metrics"
    assert stt_span.attributes["stt_status"] == "missing"
    assert "duration_ms" not in stt_span.attributes


def test_trace_waits_for_assistant_text_before_emit(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-wait-assistant",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hi", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-wait-assistant"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-wait-assistant"))
        await collector.wait_for_pending_trace_tasks()
        assert not fake_tracer.spans

        await collector.on_conversation_item_added(role="assistant", content="hello there")
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())
    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    assert turn_spans[0].attributes["langfuse.trace.output"] == "hello there"


def test_tts_metric_metadata_assistant_text_emits_without_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-tts-metadata",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hi", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-tts-metadata"))
        tts_metrics = _make_tts_metrics("speech-tts-metadata")
        tts_metrics.metadata = {
            "model_name": "pocket-tts",
            "model_provider": "Kyutai",
            "assistant_text": "assistant text from tts metadata",
        }
        await collector.on_metrics_collected(tts_metrics)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.output"] == "assistant text from tts metadata"
    assert root.attributes["langfuse.trace.metadata.assistant_text_missing"] is False
    assert root.attributes["langfuse.trace.metadata.assistant_text_source"] == "tts_metrics"
    assert root.attributes["langfuse.trace.metadata.finalization_reason"] == "complete"


def test_speech_item_added_assistant_text_arriving_within_grace_emits_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.05
    handle = _FakeSpeechHandle([], speech_id="speech-item-added")

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-speech-item-added",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hi", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-item-added"))
        await collector.on_speech_created(handle)
        await collector.on_metrics_collected(_make_tts_metrics("speech-item-added"))
        await collector.wait_for_pending_trace_tasks()
        assert not fake_tracer.spans

        handle.add_chat_item(
            _FakeChatItem(
                role="assistant",
                content=["assistant text from speech item callback"],
            )
        )
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.output"] == "assistant text from speech item callback"
    assert root.attributes["langfuse.trace.metadata.assistant_text_missing"] is False
    assert root.attributes["langfuse.trace.metadata.assistant_text_source"] == "speech_item_added"
    assert root.attributes["langfuse.trace.metadata.finalization_reason"] == "complete"


def test_speech_created_done_callback_backfills_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-speech-created-fallback",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hi there", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-speech-created"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-speech-created"))
        await collector.wait_for_pending_trace_tasks()
        assert not fake_tracer.spans

        handle = _FakeSpeechHandleWithoutItemAddedHook(
            chat_items=[_FakeChatItem(role="assistant", content=["fallback reply"])],
            speech_id="speech-speech-created",
        )
        await collector.on_speech_created(handle)
        handle.trigger_done()
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    assert turn_spans[0].attributes["langfuse.trace.output"] == "fallback reply"
    assert turn_spans[0].attributes["langfuse.trace.metadata.assistant_text_missing"] is False


def test_speech_done_does_not_replace_speech_item_added_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    handle = _FakeSpeechHandle([], speech_id="speech-speech-item-preferred")

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-speech-item-preferred",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hi there", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-speech-item-preferred"))
        await collector.on_speech_created(handle)
        handle.add_chat_item(
            _FakeChatItem(role="assistant", content=["preferred reply from speech item"])
        )
        await asyncio.sleep(0)

        handle.chat_items.append(
            _FakeChatItem(role="assistant", content=["stale reply from speech done"])
        )
        handle.trigger_done()
        await asyncio.sleep(0)

        await collector.on_metrics_collected(_make_tts_metrics("speech-speech-item-preferred"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.output"] == "preferred reply from speech item"
    assert root.attributes["langfuse.trace.metadata.assistant_text_source"] == "speech_item_added"


def test_conversation_item_added_before_speech_item_added_rescues_unique_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.2
    handle = _FakeSpeechHandle([], speech_id="speech-item-identity-first-conversation")
    item = _FakeChatItem(role="assistant", content=["reply from exact item identity"])

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-item-identity-first-conversation",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(
            _make_llm_metrics("speech-item-identity-first-conversation")
        )
        await collector.on_speech_created(handle)
        await collector.on_metrics_collected(
            _make_tts_metrics("speech-item-identity-first-conversation")
        )
        await collector.on_conversation_item_added(
            role="assistant",
            item=item,
            content=item.content,
        )
        await collector.wait_for_pending_trace_tasks()
        turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
        assert len(turn_spans) == 1
        assert turn_spans[0].attributes["langfuse.trace.output"] == "reply from exact item identity"
        assert (
            turn_spans[0].attributes["langfuse.trace.metadata.assistant_text_source"]
            == "conversation_item_safe_rescue"
        )

        handle.add_chat_item(item)
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.output"] == "reply from exact item identity"
    assert root.attributes["langfuse.trace.metadata.assistant_text_source"] in {
        "conversation_item_safe_rescue",
        "speech_item_added",
    }


def test_conversation_item_added_after_speech_item_added_uses_same_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    handle = _FakeSpeechHandle([], speech_id="speech-item-identity-first-speech")
    item = _FakeChatItem(role="assistant", content=["same item same turn"])

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-item-identity-first-speech",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(
            _make_llm_metrics("speech-item-identity-first-speech")
        )
        await collector.on_speech_created(handle)
        handle.add_chat_item(item)
        await asyncio.sleep(0)
        await collector.on_metrics_collected(
            _make_tts_metrics("speech-item-identity-first-speech")
        )
        await collector.on_conversation_item_added(
            role="assistant",
            item=item,
            content=item.content,
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.output"] == "same item same turn"
    assert root.attributes["langfuse.trace.metadata.assistant_text_source"] == "speech_item_added"


def test_late_conversation_item_from_previous_turn_does_not_shift_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.2
    handle_1 = _FakeSpeechHandle([], speech_id="speech-shift-a")
    handle_2 = _FakeSpeechHandle([], speech_id="speech-shift-b")
    item_1 = _FakeChatItem(role="assistant", content=["Hi. How can I help?"])
    item_2 = _FakeChatItem(role="assistant", content=["Focus on small, consistent actions."])

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-no-output-shift",
            participant_id="web-123",
        )

        await collector.on_user_input_transcribed("Hello.", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-shift-a"))
        await collector.on_speech_created(handle_1)
        await collector.on_metrics_collected(_make_tts_metrics("speech-shift-a"))

        await collector.on_user_input_transcribed(
            "Tell me something short but valuable.",
            is_final=True,
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-shift-b"))
        await collector.on_speech_created(handle_2)
        await collector.on_metrics_collected(_make_tts_metrics("speech-shift-b"))

        await collector.on_conversation_item_added(
            role="assistant",
            item=item_1,
            content=item_1.content,
        )
        await collector.wait_for_pending_trace_tasks()
        assert not [span for span in fake_tracer.spans if span.name == "turn"]

        handle_1.add_chat_item(item_1)
        await asyncio.sleep(0)

        await collector.on_conversation_item_added(
            role="assistant",
            item=item_2,
            content=item_2.content,
        )
        handle_2.add_chat_item(item_2)
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 2
    first = next(
        span for span in turn_spans if span.attributes["langfuse.trace.input"] == "Hello."
    )
    second = next(
        span
        for span in turn_spans
        if span.attributes["langfuse.trace.input"] == "Tell me something short but valuable."
    )
    assert first.attributes["langfuse.trace.output"] == "Hi. How can I help?"
    assert second.attributes["langfuse.trace.output"] == "Focus on small, consistent actions."
    assert first.attributes["langfuse.trace.output"] != second.attributes["langfuse.trace.output"]
    assert first.attributes["langfuse.trace.metadata.assistant_text_missing"] is False
    assert second.attributes["langfuse.trace.metadata.assistant_text_missing"] is False


def test_unresolved_conversation_item_rescues_unique_no_tool_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    item = _FakeChatItem(role="assistant", content=["rescued from unresolved item"])

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-safe-rescue",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-safe-rescue"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-safe-rescue"))
        await collector.on_conversation_item_added(
            role="assistant",
            item=item,
            content=item.content,
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.output"] == "rescued from unresolved item"
    assert (
        root.attributes["langfuse.trace.metadata.assistant_text_source"]
        == "conversation_item_safe_rescue"
    )


def test_ambiguous_unresolved_conversation_item_waits_for_exact_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.05
    collector._tracer._trace_legacy_finalize_timeout_sec = 0.2
    handle_1 = _FakeSpeechHandle([], speech_id="speech-safe-ambiguous-a")
    handle_2 = _FakeSpeechHandle([], speech_id="speech-safe-ambiguous-b")
    item_1 = _FakeChatItem(role="assistant", content=["first exact reply"])
    item_2 = _FakeChatItem(role="assistant", content=["second exact reply"])

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-safe-ambiguous",
            participant_id="web-123",
        )

        await collector.on_user_input_transcribed("first input", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-safe-ambiguous-a"))
        await collector.on_speech_created(handle_1)
        await collector.on_metrics_collected(_make_tts_metrics("speech-safe-ambiguous-a"))

        await collector.on_user_input_transcribed("second input", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-safe-ambiguous-b"))
        await collector.on_speech_created(handle_2)
        await collector.on_metrics_collected(_make_tts_metrics("speech-safe-ambiguous-b"))

        await collector.on_conversation_item_added(
            role="assistant",
            item=item_1,
            content=item_1.content,
        )
        await collector.wait_for_pending_trace_tasks()
        assert not [span for span in fake_tracer.spans if span.name == "turn"]

        handle_1.add_chat_item(item_1)
        handle_2.add_chat_item(item_2)
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 2
    first = next(
        span
        for span in turn_spans
        if span.attributes["langfuse.trace.input"] == "first input"
    )
    second = next(
        span
        for span in turn_spans
        if span.attributes["langfuse.trace.input"] == "second input"
    )
    assert first.attributes["langfuse.trace.output"] == "first exact reply"
    assert second.attributes["langfuse.trace.output"] == "second exact reply"
    assert (
        first.attributes["langfuse.trace.metadata.assistant_text_source"]
        == "speech_item_added"
    )
    assert (
        second.attributes["langfuse.trace.metadata.assistant_text_source"]
        == "speech_item_added"
    )


def test_exact_assistant_text_upgrades_safe_rescue_source() -> None:
    room = _FakeRoom()
    tracer = TurnTracer(
        publisher=ChannelPublisher(room),  # type: ignore[arg-type]
        room_id="RM123",
        session_id="session-safe-rescue-upgrade",
        participant_id="web-123",
        fallback_session_id=None,
        fallback_participant_id=None,
        langfuse_enabled=False,
        pending_agent_transcripts=deque(),
        pending_assistant_items={},
    )

    async def _run() -> None:
        await tracer.create_turn(user_transcript="hello", room_id="RM123")
        turn = await tracer.attach_llm(
            duration=0.6,
            ttft=0.1,
            speech_id="speech-safe-rescue-upgrade",
        )
        assert turn is not None
        turn = await tracer.attach_tts(
            duration=0.5,
            fallback_duration=0.5,
            ttfb=0.15,
            speech_id="speech-safe-rescue-upgrade",
            observed_total_latency=None,
        )
        assert turn is not None
        tracer._pending_assistant_items[123] = PendingAssistantItemRecord(
            item_id=123,
            text="rescued text",
            source="conversation_item",
            observed_at=time.time(),
        )

        rescued_turn = await tracer.try_attach_unresolved_assistant_item(123)
        assert rescued_turn is turn
        assert turn.assistant_text == "rescued text"
        assert turn.assistant_text_source == "conversation_item_safe_rescue"

        upgraded_turn = await tracer.attach_assistant_text(
            "rescued text",
            speech_id="speech-safe-rescue-upgrade",
            source="speech_item_added",
        )
        assert upgraded_turn is turn
        assert turn.assistant_text == "rescued text"
        assert turn.assistant_text_source == "speech_item_added"

    asyncio.run(_run())


def test_unscoped_streamed_text_flush_rescues_unique_no_tool_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-unscoped-stream",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-unscoped-stream"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-unscoped-stream"))
        collector.submit_streamed_assistant_text_delta(
            None,
            "hello from streamed output",
            time.time(),
        )
        collector.submit_streamed_assistant_text_flush(None, time.time())
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.output"] == "hello from streamed output"
    assert root.attributes["langfuse.trace.metadata.assistant_text_source"] == "text_output_stream"
    assert root.attributes["langfuse.trace.metadata.streamed_assistant_text_flushed"] is True


def test_unscoped_streamed_text_does_not_finalize_before_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-unscoped-no-flush",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-unscoped-no-flush"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-unscoped-no-flush"))
        collector.submit_streamed_assistant_text_delta(
            None,
            "partial streamed output",
            time.time(),
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    assert not [span for span in fake_tracer.spans if span.name == "turn"]


def test_unscoped_streamed_text_upgrades_to_exact_source() -> None:
    room = _FakeRoom()
    pending_unscoped_records: deque[PendingUnscopedStreamRecord] = deque()
    tracer = TurnTracer(
        publisher=ChannelPublisher(room),  # type: ignore[arg-type]
        room_id="RM123",
        session_id="session-stream-upgrade",
        participant_id="web-123",
        fallback_session_id=None,
        fallback_participant_id=None,
        langfuse_enabled=False,
        pending_agent_transcripts=deque(),
        pending_assistant_items={},
        pending_unscoped_stream_records=pending_unscoped_records,
    )

    async def _run() -> None:
        await tracer.create_turn(user_transcript="hello", room_id="RM123")
        turn = await tracer.attach_llm(
            duration=0.6,
            ttft=0.1,
            speech_id="speech-stream-upgrade",
        )
        assert turn is not None
        turn = await tracer.attach_tts(
            duration=0.5,
            fallback_duration=0.5,
            ttfb=0.15,
            speech_id="speech-stream-upgrade",
            observed_total_latency=None,
        )
        assert turn is not None
        pending_unscoped_records.append(
            PendingUnscopedStreamRecord(
                text="rescued streamed text",
                observed_at=time.time(),
                last_delta_at=time.time(),
                flush_observed=True,
                flush_observed_at=time.time(),
            )
        )

        rescued_turn = await tracer.try_attach_unscoped_streamed_assistant_text()
        assert rescued_turn is turn
        assert turn.assistant_text == "rescued streamed text"
        assert turn.assistant_text_source == "text_output_stream"

        upgraded_turn = await tracer.attach_assistant_text(
            "rescued streamed text",
            speech_id="speech-stream-upgrade",
            source="speech_item_added",
        )
        assert upgraded_turn is turn
        assert turn.assistant_text == "rescued streamed text"
        assert turn.assistant_text_source == "speech_item_added"

    asyncio.run(_run())


def test_trace_finalize_timeout_uses_unscoped_streamed_text_before_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01
    collector._tracer._trace_legacy_finalize_timeout_sec = 0.05

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-unscoped-timeout",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-unscoped-timeout"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-unscoped-timeout"))
        collector.submit_streamed_assistant_text_delta(
            None,
            "rescued before placeholder",
            time.time(),
        )
        await asyncio.sleep(0.08)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.output"] == "rescued before placeholder"
    assert root.attributes["langfuse.trace.metadata.assistant_text_source"] == "text_output_stream"
    assert root.attributes["langfuse.trace.metadata.finalization_reason"] == "speech_done_timeout"


def test_speech_created_immediate_capture_backfills_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-speech-created-immediate",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hi there", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-speech-created-immediate"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-speech-created-immediate"))
        await collector.wait_for_pending_trace_tasks()
        assert not fake_tracer.spans

        handle = _FakeSpeechHandle(
            chat_items=[
                _FakeChatItem(
                    role="assistant",
                    content=[_FakeTextMethodPart("immediate fallback reply")],
                )
            ],
            speech_id="speech-speech-created-immediate",
        )
        await collector.on_speech_created(handle)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    assert turn_spans[0].attributes["langfuse.trace.output"] == "immediate fallback reply"
    assert turn_spans[0].attributes["langfuse.trace.metadata.assistant_text_missing"] is False


def test_regular_turn_waits_for_speech_done_before_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01
    collector._tracer._trace_legacy_finalize_timeout_sec = 0.05
    handle = _FakeSpeechHandle([], speech_id="speech-waits-for-done")

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-waits-for-speech-done",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-waits-for-done"))
        await collector.on_speech_created(handle)
        await collector.on_metrics_collected(_make_tts_metrics("speech-waits-for-done"))
        await asyncio.sleep(0.02)
        await collector.wait_for_pending_trace_tasks()
        assert not [span for span in fake_tracer.spans if span.name == "turn"]

        handle.chat_items.append(
            _FakeChatItem(role="assistant", content=["hello from speech done"])
        )
        handle.trigger_done()
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.output"] == "hello from speech done"
    assert root.attributes["langfuse.trace.metadata.assistant_text_missing"] is False
    assert root.attributes["langfuse.trace.metadata.assistant_text_source"] == "speech_done"
    assert root.attributes["langfuse.trace.metadata.speech_done_observed"] is True
    assert root.attributes["langfuse.trace.metadata.finalization_reason"] == "complete"


def test_missing_item_added_hook_rescues_unique_turn_before_speech_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01
    collector._tracer._trace_legacy_finalize_timeout_sec = 0.05
    handle = _FakeSpeechHandleWithoutItemAddedHook([], speech_id="speech-done-correlation")
    item = _FakeChatItem(role="assistant", content=["reply rescued at speech done"])

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-speech-done-correlation",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hello", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-done-correlation"))
        await collector.on_speech_created(handle)
        await collector.on_metrics_collected(_make_tts_metrics("speech-done-correlation"))
        await collector.on_conversation_item_added(
            role="assistant",
            item=item,
            content=item.content,
        )
        await asyncio.sleep(0.02)
        await collector.wait_for_pending_trace_tasks()
        turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
        assert len(turn_spans) == 1
        assert turn_spans[0].attributes["langfuse.trace.output"] == "reply rescued at speech done"
        assert (
            turn_spans[0].attributes["langfuse.trace.metadata.assistant_text_source"]
            == "conversation_item_safe_rescue"
        )

        handle.chat_items.append(item)
        handle.trigger_done()
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.output"] == "reply rescued at speech done"
    assert root.attributes["langfuse.trace.metadata.assistant_text_missing"] is False
    assert root.attributes["langfuse.trace.metadata.assistant_text_source"] in {
        "conversation_item_safe_rescue",
        "speech_done",
    }


def test_trace_finalize_timeout_for_missing_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01
    collector._tracer._trace_legacy_finalize_timeout_sec = 0.03

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-assistant-timeout",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hi", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-timeout"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-timeout"))
        await asyncio.sleep(0.015)
        await collector.wait_for_pending_trace_tasks()
        assert not [span for span in fake_tracer.spans if span.name == "turn"]

        await asyncio.sleep(0.03)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.metadata.assistant_text_missing"] is True
    assert root.attributes["langfuse.trace.output"] == "[assistant text unavailable]"
    assert root.attributes["langfuse.trace.metadata.assistant_text_source"] == "unavailable"
    assert root.attributes["langfuse.trace.metadata.speech_done_observed"] is False
    assert root.attributes["langfuse.trace.metadata.finalization_reason"] == "speech_done_timeout"


def test_trace_finalize_timeout_uses_safe_rescue_after_ambiguity_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01
    collector._tracer._trace_legacy_finalize_timeout_sec = 0.05
    handle_1 = _FakeSpeechHandle([], speech_id="speech-timeout-safe-a")
    unresolved_item = _FakeChatItem(
        role="assistant",
        content=["rescued after ambiguity clears"],
    )
    exact_item = _FakeChatItem(role="assistant", content=["exact first reply"])

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-timeout-safe-rescue",
            participant_id="web-123",
        )

        await collector.on_user_input_transcribed("first input", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-timeout-safe-a"))
        await collector.on_speech_created(handle_1)
        await collector.on_metrics_collected(_make_tts_metrics("speech-timeout-safe-a"))

        await collector.on_user_input_transcribed("second input", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-timeout-safe-b"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-timeout-safe-b"))

        await collector.on_conversation_item_added(
            role="assistant",
            item=unresolved_item,
            content=unresolved_item.content,
        )
        await collector.wait_for_pending_trace_tasks()
        assert not [span for span in fake_tracer.spans if span.name == "turn"]

        handle_1.add_chat_item(exact_item)
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

        await asyncio.sleep(0.06)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 2
    first = next(
        span
        for span in turn_spans
        if span.attributes["langfuse.trace.input"] == "first input"
    )
    second = next(
        span
        for span in turn_spans
        if span.attributes["langfuse.trace.input"] == "second input"
    )
    assert first.attributes["langfuse.trace.output"] == "exact first reply"
    assert (
        second.attributes["langfuse.trace.output"]
        == "rescued after ambiguity clears"
    )
    assert (
        second.attributes["langfuse.trace.metadata.assistant_text_source"]
        == "conversation_item_safe_rescue"
    )
    assert second.attributes["langfuse.trace.metadata.assistant_text_missing"] is False


def test_trace_finalize_timeout_uses_pending_assistant_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01
    collector._tracer._trace_legacy_finalize_timeout_sec = 0.03

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-assistant-timeout-pending-transcript",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hi", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-timeout-pending-transcript"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-timeout-pending-transcript"))

        collector._pending_agent_transcripts.append("queued assistant fallback")
        await asyncio.sleep(0.05)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.metadata.assistant_text_missing"] is False
    assert root.attributes["langfuse.trace.output"] == "queued assistant fallback"


def test_long_response_latency_accounts_for_llm_to_tts_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        speech_id = "speech-long-gap"
        await collector.on_session_metadata(
            session_id="session-long-gap",
            participant_id="web-123",
        )
        await collector.on_speech_created(_FakeSpeechHandle(chat_items=[], speech_id=speech_id))
        await collector.on_user_input_transcribed("Explain neural networks", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics(speech_id, delay=0.0, transcription_delay=0.2)
        )
        await collector.on_metrics_collected(
            _make_llm_metrics(speech_id, duration=2.0, ttft=0.01)
        )
        await collector.on_conversation_item_added(
            role="assistant",
            content="A neural network is a layered function approximator.",
        )
        await asyncio.sleep(0.2)
        await collector.on_agent_state_changed(old_state="thinking", new_state="speaking")
        await collector.on_metrics_collected(
            _make_tts_metrics(speech_id, ttfb=0.01, duration=0.2, audio_duration=0.8)
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    span_names = [span.name for span in fake_tracer.spans]
    assert span_names.index("STTMetrics") < span_names.index("EOUMetrics")
    assert span_names.index("agent_response_phase_1") < span_names.index("LLMMetrics")
    assert span_names.index("LLMMetrics") < span_names.index("TTSMetrics")
    assert span_names.index("TTSMetrics") < span_names.index("metrics_summary")
    if "perceived_latency_first_audio" in span_names:
        assert span_names.index("metrics_summary") < span_names.index("perceived_latency_first_audio")

    assert root.attributes["latency_ms.perceived_first_audio"] > 200.0
    assert root.attributes["latency_ms.conversational"] > 200.0
    assert root.attributes["latency_ms.llm_to_tts_handoff"] > 150.0
    assert root.attributes["latency_ms.conversational"] == pytest.approx(
        root.attributes["latency_ms.eou_delay"]
        + root.attributes["latency_ms.llm_ttft"]
        + root.attributes["latency_ms.llm_to_tts_handoff"]
        + root.attributes["latency_ms.tts_ttfb"],
        abs=5.0,
    )
    assert root.attributes["latency_ms.stt_finalization"] == pytest.approx(200.0)

    payloads = _decode_payloads(room)
    conversation_turns = [
        payload for payload in payloads if payload.get("type") == "conversation_turn"
    ]
    agent_turn = next(payload for payload in conversation_turns if payload.get("role") == "agent")
    assert agent_turn["latencies"]["total_latency"] == pytest.approx(
        root.attributes["latency_ms.conversational"] / 1000.0,
        abs=0.05,
    )
    assert agent_turn["latencies"]["llm_to_tts_handoff_latency"] == pytest.approx(
        root.attributes["latency_ms.llm_to_tts_handoff"] / 1000.0,
        abs=0.05,
    )


def test_fallback_console_session_id_is_used_when_metadata_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="console-user",
        fallback_session_prefix="console",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_user_input_transcribed("console test", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-console"))
        await collector.on_conversation_item_added(role="assistant", content="console reply")
        await collector.on_metrics_collected(_make_tts_metrics("speech-console"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    session_id = turn_spans[0].attributes["session_id"]
    assert session_id.startswith("console_")
    assert session_id != "unknown-session"


def test_real_session_metadata_overrides_fallback_for_pending_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="console-user",
        fallback_session_prefix="console",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_user_input_transcribed("override test", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-override"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-override"))
        await collector.on_session_metadata(
            session_id="session-real",
            participant_id="web-override",
        )
        await collector.on_conversation_item_added(role="assistant", content="reply")
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    assert turn_spans[0].attributes["session_id"] == "session-real"


def test_fallback_console_participant_id_is_used_when_metadata_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id=None,
        fallback_session_prefix="console",
        fallback_participant_prefix="console",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_user_input_transcribed("console participant", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-console-participant"))
        await collector.on_conversation_item_added(role="assistant", content="ok")
        await collector.on_metrics_collected(_make_tts_metrics("speech-console-participant"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    participant_id = turn_spans[0].attributes["participant_id"]
    assert participant_id.startswith("console_")
    assert participant_id != "unknown-participant"


def test_real_participant_metadata_overrides_fallback_for_pending_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id=None,
        fallback_session_prefix="console",
        fallback_participant_prefix="console",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_user_input_transcribed("override participant", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-override-participant"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-override-participant"))
        await collector.on_session_metadata(
            session_id="session-real-participant",
            participant_id="web-real-participant",
        )
        await collector.on_conversation_item_added(role="assistant", content="reply")
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    assert turn_spans[0].attributes["session_id"] == "session-real-participant"
    assert turn_spans[0].attributes["participant_id"] == "web-real-participant"


def test_stale_orphan_assistant_text_from_absorbed_turn_is_not_attached_to_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    base_time = time.time()
    final_input = "What is the difference between speech to text and speech recognition?"
    final_output = (
        "Speech recognition detects spoken words. Speech to text writes them down."
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-stale-orphan-after-coalesce",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("What", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-a", delay=0.7, transcription_delay=0.2)
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-a"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-a"))

        await collector.on_user_input_transcribed(
            "is the difference between speech to text and speech recognition?",
            is_final=True,
        )
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-b", delay=0.7, transcription_delay=0.2)
        )
        await collector.on_conversation_item_added(
            role="user",
            content=final_input,
            event_created_at=base_time + 0.25,
            item_created_at=base_time + 0.25,
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-b"))

        await collector.on_conversation_item_added(
            role="assistant",
            content="stale reply from the absorbed turn",
            event_created_at=base_time + 1.0,
            item_created_at=base_time + 0.1,
        )
        await collector.on_conversation_item_added(
            role="assistant",
            content=final_output,
            event_created_at=base_time + 1.2,
            item_created_at=base_time + 1.2,
        )
        await collector.on_metrics_collected(_make_tts_metrics("speech-b"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.input"] == final_input
    assert root.attributes["langfuse.trace.output"] == final_output
    assert "stale reply from the absorbed turn" not in root.attributes["langfuse.trace.output"]


def test_audio_started_turn_waits_for_timeout_before_placeholder_on_barge_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.01

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-audio-started-no-coalesce",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("first prompt", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-audio-started", delay=0.4, transcription_delay=0.1)
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-audio-started"))
        await collector.on_speech_created(
            _FakeSpeechHandle(chat_items=[], speech_id="speech-audio-started")
        )
        await collector.on_agent_state_changed(
            old_state="thinking",
            new_state="speaking",
        )
        await collector.on_metrics_collected(_make_tts_metrics("speech-audio-started"))

        await collector.on_user_input_transcribed("second prompt", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-separate-b", delay=0.5, transcription_delay=0.1)
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-separate-b"))
        await collector.on_conversation_item_added(
            role="assistant",
            content="second reply",
        )
        await collector.on_metrics_collected(_make_tts_metrics("speech-separate-b"))
        await collector.wait_for_pending_trace_tasks()

        turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
        assert len(turn_spans) == 1
        assert turn_spans[0].attributes["langfuse.trace.input"] == "second prompt"

        await asyncio.sleep(0.03)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 2
    first = next(
        span
        for span in turn_spans
        if span.attributes["langfuse.trace.input"] == "first prompt"
    )
    second = next(
        span
        for span in turn_spans
        if span.attributes["langfuse.trace.input"] == "second prompt"
    )
    assert first.attributes["langfuse.trace.metadata.interrupted"] is True
    assert first.attributes["langfuse.trace.output"] == "[assistant text unavailable]"
    assert second.attributes["langfuse.trace.output"] == "second reply"


def test_audio_started_turn_uses_late_speech_done_text_after_barge_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )
    collector._trace_finalize_timeout_sec = 0.05
    handle = _FakeSpeechHandle(chat_items=[], speech_id="speech-audio-started-late-done")

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-audio-started-late-done",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("first prompt", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics(
                "speech-audio-started-late-done",
                delay=0.4,
                transcription_delay=0.1,
            )
        )
        await collector.on_metrics_collected(
            _make_llm_metrics("speech-audio-started-late-done")
        )
        await collector.on_speech_created(handle)
        await collector.on_agent_state_changed(
            old_state="thinking",
            new_state="speaking",
        )
        await collector.on_metrics_collected(
            _make_tts_metrics("speech-audio-started-late-done")
        )

        await collector.on_user_input_transcribed("second prompt", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics("speech-separate-late-done-b", delay=0.5, transcription_delay=0.1)
        )
        await collector.on_metrics_collected(_make_llm_metrics("speech-separate-late-done-b"))
        await collector.on_conversation_item_added(
            role="assistant",
            content="second reply",
        )
        await collector.on_metrics_collected(_make_tts_metrics("speech-separate-late-done-b"))
        await collector.wait_for_pending_trace_tasks()

        turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
        assert len(turn_spans) == 1
        assert turn_spans[0].attributes["langfuse.trace.input"] == "second prompt"

        handle.chat_items = [
            _FakeChatItem(role="assistant", content=["first reply recovered late"])
        ]
        handle.trigger_done()
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 2
    first = next(
        span
        for span in turn_spans
        if span.attributes["langfuse.trace.input"] == "first prompt"
    )
    second = next(
        span
        for span in turn_spans
        if span.attributes["langfuse.trace.input"] == "second prompt"
    )
    assert first.attributes["langfuse.trace.metadata.interrupted"] is True
    assert first.attributes["langfuse.trace.output"] == "first reply recovered late"
    assert second.attributes["langfuse.trace.output"] == "second reply"


def test_interrupted_pretool_leadin_keeps_own_output_and_next_turn_stays_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-interrupted-pretool-leadin",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("search papers", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-pretool-interrupted"))
        await collector.on_tool_step_started()
        await collector.on_conversation_item_added(
            role="assistant",
            content="Let me check that.",
        )
        await collector.on_speech_created(
            _FakeSpeechHandle(chat_items=[], speech_id="speech-pretool-interrupted")
        )
        await collector.on_agent_state_changed(
            old_state="thinking",
            new_state="speaking",
        )
        await collector.on_metrics_collected(_make_tts_metrics("speech-pretool-interrupted"))

        await collector.on_user_input_transcribed("never mind", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-pretool-next"))
        await collector.on_conversation_item_added(
            role="assistant",
            content="Okay, stopping here.",
        )
        await collector.on_metrics_collected(_make_tts_metrics("speech-pretool-next"))
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 2
    first, second = turn_spans
    assert first.attributes["langfuse.trace.input"] == "search papers"
    assert first.attributes["langfuse.trace.metadata.interrupted"] is True
    assert first.attributes["langfuse.trace.output"] == "Let me check that."
    assert second.attributes["langfuse.trace.input"] == "never mind"
    assert second.attributes["langfuse.trace.output"] == "Okay, stopping here."


def test_drain_pending_traces_finalizes_without_manual_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

    room = _FakeRoom()
    collector = MetricsCollector(
        room=room,  # type: ignore[arg-type]
        model_name="moonshine",
        room_name=room.name,
        room_id="RM123",
        participant_id="web-123",
        langfuse_enabled=True,
    )

    async def _run() -> None:
        await collector.on_session_metadata(
            session_id="session-drain-pending-traces",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("drain pending", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-drain-pending"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-drain-pending"))
        await collector.drain_pending_traces()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.input"] == "drain pending"
    assert root.attributes["langfuse.trace.output"] == "[assistant text unavailable]"
    assert root.attributes["langfuse.trace.metadata.interrupted"] is True
    assert root.attributes["langfuse.trace.metadata.interrupted_reason"] == "shutdown_drain"
