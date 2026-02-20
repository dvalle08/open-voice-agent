from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

from opentelemetry import trace as otel_trace
from livekit.agents import metrics

from src.agent.metrics_collector import MetricsCollector


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


@dataclass
class _FakeChatItem:
    role: str
    content: list[Any]


class _FakeTextMethodPart:
    def __init__(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text


class _FakeSpeechHandle:
    def __init__(self, chat_items: list[Any]) -> None:
        self.chat_items = chat_items
        self._callbacks: list[Any] = []

    def add_done_callback(self, callback: Any) -> None:
        self._callbacks.append(callback)

    def trigger_done(self) -> None:
        for callback in self._callbacks:
            callback(self)


def test_turn_trace_has_required_metadata_and_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agent.metrics_collector as metrics_collector_module

    fake_tracer = _FakeTracer()
    monkeypatch.setattr(metrics_collector_module, "tracer", fake_tracer)

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
    assert span_names == ["turn", "user_input", "vad", "stt", "llm", "tts", "conversation_latency"]

    root, user_input_span, vad_span, stt_span, llm_span, tts_span, conversational_span = fake_tracer.spans
    assert root.attributes["session_id"] == "session-abc"
    assert root.attributes["room_id"] == "RM123"
    assert root.attributes["participant_id"] == "web-123"
    assert root.attributes["turn_id"]
    assert root.attributes["langfuse.trace.output"] == "hi, how can I help?"
    assert root.attributes["latency_ms.eou_delay"] == pytest.approx(1100.0)
    assert root.attributes["latency_ms.stt_finalization"] == pytest.approx(250.0)
    assert root.attributes["latency_ms.stt_total"] == pytest.approx(1350.0)
    assert root.attributes["latency_ms.llm_ttft"] > 0
    assert root.attributes["latency_ms.llm_total"] > 0
    assert root.attributes["latency_ms.tts_ttfb"] > 0
    assert root.attributes["latency_ms.conversational"] == pytest.approx(1600.0)
    assert root.attributes["latency_ms.speech_end_to_assistant_speech_start"] == pytest.approx(1600.0)

    assert user_input_span.attributes["user_transcript"] == "hello there"
    assert vad_span.attributes["duration_ms"] == pytest.approx(1100.0)

    assert stt_span.attributes["user_transcript"] == "hello there"
    assert stt_span.attributes["stt_status"] == "measured"
    assert stt_span.attributes["duration_ms"] == pytest.approx(1350.0)
    assert stt_span.attributes["stt_finalization_ms"] == pytest.approx(250.0)
    assert stt_span.attributes["stt_total_latency_ms"] == pytest.approx(1350.0)

    assert llm_span.attributes["prompt_text"] == "hello there"
    assert llm_span.attributes["response_text"] == "hi, how can I help?"
    assert llm_span.attributes["ttft_ms"] > 0
    assert llm_span.attributes["llm_total_latency_ms"] > 0
    assert llm_span.attributes["input"] == "hello there"
    assert llm_span.attributes["output"] == "hi, how can I help?"
    assert llm_span.attributes["duration_ms"] > 0

    assert tts_span.attributes["assistant_text"] == "hi, how can I help?"
    assert tts_span.attributes["ttfb_ms"] > 0
    assert tts_span.attributes["input"] == "hi, how can I help?"
    assert tts_span.attributes["output"] == "hi, how can I help?"
    assert tts_span.attributes["duration_ms"] > 0

    assert conversational_span.attributes["duration_ms"] == pytest.approx(1600.0)
    assert (
        conversational_span.attributes["speech_end_to_assistant_speech_start_ms"]
        == pytest.approx(1600.0)
    )
    assert conversational_span.attributes["eou_delay_ms"] == pytest.approx(1100.0)
    assert conversational_span.attributes["stt_finalization_ms"] == pytest.approx(250.0)
    assert conversational_span.attributes["llm_ttft_ms"] > 0
    assert conversational_span.attributes["tts_ttfb_ms"] > 0
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


def test_tracing_failure_does_not_break_metrics_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agent.metrics_collector as metrics_collector_module

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
    import src.agent.metrics_collector as metrics_collector_module

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


def test_trace_emits_without_stt_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agent.metrics_collector as metrics_collector_module

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
    assert span_names == ["turn", "user_input", "vad", "stt", "llm", "tts"]
    stt_span = fake_tracer.spans[3]
    assert stt_span.attributes["user_transcript"] == "turn without stt metrics"
    assert stt_span.attributes["stt_status"] == "missing"
    assert "duration_ms" not in stt_span.attributes


def test_trace_waits_for_assistant_text_before_emit(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agent.metrics_collector as metrics_collector_module

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


def test_speech_created_done_callback_backfills_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.metrics_collector as metrics_collector_module

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

        handle = _FakeSpeechHandle(chat_items=[_FakeChatItem(role="assistant", content=["fallback reply"])])
        await collector.on_speech_created(handle)
        handle.trigger_done()
        await asyncio.sleep(0)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    assert turn_spans[0].attributes["langfuse.trace.output"] == "fallback reply"
    assert turn_spans[0].attributes["langfuse.trace.metadata.assistant_text_missing"] is False


def test_speech_created_immediate_capture_backfills_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.metrics_collector as metrics_collector_module

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
            ]
        )
        await collector.on_speech_created(handle)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    assert turn_spans[0].attributes["langfuse.trace.output"] == "immediate fallback reply"
    assert turn_spans[0].attributes["langfuse.trace.metadata.assistant_text_missing"] is False


def test_trace_finalize_timeout_for_missing_assistant_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.metrics_collector as metrics_collector_module

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
            session_id="session-assistant-timeout",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hi", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-timeout"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-timeout"))
        await asyncio.sleep(0.03)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.metadata.assistant_text_missing"] is True
    assert root.attributes["langfuse.trace.output"] == "[assistant text unavailable]"


def test_trace_finalize_timeout_uses_pending_assistant_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.metrics_collector as metrics_collector_module

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
            session_id="session-assistant-timeout-pending-transcript",
            participant_id="web-123",
        )
        await collector.on_user_input_transcribed("hi", is_final=True)
        await collector.on_metrics_collected(_make_llm_metrics("speech-timeout-pending-transcript"))
        await collector.on_metrics_collected(_make_tts_metrics("speech-timeout-pending-transcript"))

        collector._pending_agent_transcripts.append("queued assistant fallback")
        await asyncio.sleep(0.03)
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]
    assert root.attributes["langfuse.trace.metadata.assistant_text_missing"] is False
    assert root.attributes["langfuse.trace.output"] == "queued assistant fallback"


def test_long_response_latency_accounts_for_llm_generation_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.metrics_collector as metrics_collector_module

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
        await collector.on_user_input_transcribed("Explain neural networks", is_final=True)
        await collector.on_metrics_collected(
            _make_eou_metrics(speech_id, delay=0.0, transcription_delay=0.0)
        )
        await collector.on_metrics_collected(
            _make_llm_metrics(speech_id, duration=2.0, ttft=0.01)
        )
        await collector.on_conversation_item_added(
            role="assistant",
            content="A neural network is a layered function approximator.",
        )
        await asyncio.sleep(0.2)
        await collector.on_metrics_collected(
            _make_tts_metrics(speech_id, ttfb=0.01, duration=0.2, audio_duration=0.8)
        )
        await collector.wait_for_pending_trace_tasks()

    asyncio.run(_run())

    turn_spans = [span for span in fake_tracer.spans if span.name == "turn"]
    assert len(turn_spans) == 1
    root = turn_spans[0]

    assert root.attributes["latency_ms.conversational"] > 200.0
    assert root.attributes["latency_ms.llm_generation_wait"] > 150.0
    assert root.attributes["latency_ms.conversational"] == pytest.approx(
        root.attributes["latency_ms.eou_delay"]
        + root.attributes["latency_ms.stt_finalization"]
        + root.attributes["latency_ms.llm_ttft"]
        + root.attributes["latency_ms.llm_generation_wait"]
        + root.attributes["latency_ms.tts_ttfb"],
        abs=5.0,
    )

    gap_spans = [span for span in fake_tracer.spans if span.name == "llm_generation_wait"]
    assert len(gap_spans) == 1
    assert gap_spans[0].attributes["duration_ms"] == pytest.approx(
        root.attributes["latency_ms.llm_generation_wait"],
        abs=5.0,
    )

    payloads = _decode_payloads(room)
    conversation_turns = [
        payload for payload in payloads if payload.get("type") == "conversation_turn"
    ]
    agent_turn = next(payload for payload in conversation_turns if payload.get("role") == "agent")
    assert agent_turn["latencies"]["total_latency"] == pytest.approx(
        root.attributes["latency_ms.conversational"] / 1000.0,
        abs=0.05,
    )
    assert agent_turn["latencies"]["llm_generation_wait_latency"] == pytest.approx(
        root.attributes["latency_ms.llm_generation_wait"] / 1000.0,
        abs=0.05,
    )


def test_fallback_console_session_id_is_used_when_metadata_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.metrics_collector as metrics_collector_module

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
    import src.agent.metrics_collector as metrics_collector_module

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
    import src.agent.metrics_collector as metrics_collector_module

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
    import src.agent.metrics_collector as metrics_collector_module

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
