from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest
from livekit.agents import llm

from src.agent.models.llm_runtime import (
    MCP_GENERATE_REPLY_BLOCK_MESSAGE,
    build_llm_runtime,
    build_mcp_http_timeout,
    install_mcp_generate_reply_guard,
    resolve_mcp_runtime_mode,
    run_startup_greeting,
)
from src.agent.prompts.assistant import ASSISTANT_INSTRUCTIONS
from src.agent.prompts.runtime import MCP_STARTUP_GREETING
from src.agent.runtime.tasks import (
    cancel_task_for_shutdown,
    monitor_startup_greeting_handle,
    run_llm_warmup,
    schedule_llm_warmup_task,
    schedule_startup_greeting_task,
)
from src.agent.tools.pre_tool_feedback import inject_pre_tool_feedback


class _FakeSpeechHandle:
    def __init__(self, *, speech_id: str = "speech-greeting", block: bool = False) -> None:
        self.id = speech_id
        self.interrupt_calls: list[bool] = []
        self._done = asyncio.Event()
        if not block:
            self._done.set()

    async def wait_for_playout(self) -> None:
        await self._done.wait()

    def interrupt(self, *, force: bool = False) -> "_FakeSpeechHandle":
        self.interrupt_calls.append(force)
        self._done.set()
        return self


class _FakeSession:
    def __init__(self, *, greeting_handle: _FakeSpeechHandle | None = None) -> None:
        self.say_calls: list[dict[str, object]] = []
        self.generate_reply_calls: list[dict[str, object]] = []
        self.greeting_handle = greeting_handle or _FakeSpeechHandle()

    def say(self, text: str, **kwargs: object) -> _FakeSpeechHandle:
        self.say_calls.append({"text": text, "kwargs": kwargs})
        return self.greeting_handle

    def generate_reply(self, **kwargs: object) -> str:
        self.generate_reply_calls.append(kwargs)
        return "reply-handle"


class _FailingSaySession(_FakeSession):
    def say(self, text: str, **kwargs: object) -> _FakeSpeechHandle:
        self.say_calls.append({"text": text, "kwargs": kwargs})
        raise RuntimeError("say failed")


class _FakeLLMStream:
    def __init__(
        self,
        *,
        chunks: list[object] | None = None,
        iter_error: Exception | None = None,
        block_forever: bool = False,
    ) -> None:
        self._chunks = list(chunks or [])
        self._iter_error = iter_error
        self._block_forever = block_forever
        self._idx = 0
        self._blocked = asyncio.Event()
        self.aclose_calls = 0
        self.iterations = 0
        self.closed = False

    def __aiter__(self) -> "_FakeLLMStream":
        return self

    async def __anext__(self) -> object:
        if self._block_forever:
            await self._blocked.wait()
        if self._iter_error is not None:
            raise self._iter_error
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        self.iterations += 1
        return chunk

    async def aclose(self) -> None:
        self.closed = True
        self.aclose_calls += 1
        self._blocked.set()


class _FakeLLMClient:
    def __init__(self, stream: _FakeLLMStream) -> None:
        self._stream = stream
        self.chat_calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> _FakeLLMStream:
        self.chat_calls.append(kwargs)
        return self._stream


class _FakeToolFeedback:
    def __init__(self, *, fallback_phrase: str = "Let me check that.") -> None:
        self.fallback_phrase = fallback_phrase
        self.start_typing_calls = 0

    def next_fallback_phrase(self) -> str:
        return self.fallback_phrase

    async def start_typing_sound(self) -> None:
        self.start_typing_calls += 1


class _FakeToolStepStartedCallback:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


async def _iter_items(items: list[Any]) -> Any:
    for item in items:
        yield item


async def _collect(items: Any) -> list[Any]:
    out: list[Any] = []
    async for item in items:
        out.append(item)
    return out


def _tool_chunk(*, content: str | None, call_id: str = "call-1") -> llm.ChatChunk:
    return llm.ChatChunk(
        id="chunk-1",
        delta=llm.ChoiceDelta(
            content=content,
            tool_calls=[
                llm.FunctionToolCall(
                    name="paper_search",
                    arguments='{"query":"transformers"}',
                    call_id=call_id,
                )
            ],
        ),
    )


def test_resolve_mcp_runtime_mode_enables_mcp_with_supported_config() -> None:
    decision = resolve_mcp_runtime_mode(
        mcp_enabled=True,
        llm_provider="NVIDIA",
        nvidia_api_key="nvapi-test",
    )

    assert decision.enabled is True
    assert decision.reason == "mcp_enabled"


def test_resolve_mcp_runtime_mode_respects_disabled_flag() -> None:
    decision = resolve_mcp_runtime_mode(
        mcp_enabled=False,
        llm_provider="nvidia",
        nvidia_api_key="nvapi-test",
    )

    assert decision.enabled is False
    assert decision.reason == "mcp_disabled"


def test_resolve_mcp_runtime_mode_rejects_unsupported_provider() -> None:
    decision = resolve_mcp_runtime_mode(
        mcp_enabled=True,
        llm_provider="unknown-provider",
        nvidia_api_key="nvapi-test",
    )

    assert decision.enabled is False
    assert decision.reason == "provider_not_supported:unknown-provider"


def test_resolve_mcp_runtime_mode_supports_ollama_provider() -> None:
    decision = resolve_mcp_runtime_mode(
        mcp_enabled=True,
        llm_provider="ollama",
        nvidia_api_key=None,
    )

    assert decision.enabled is True
    assert decision.reason == "mcp_enabled"


def test_resolve_mcp_runtime_mode_requires_nvidia_api_key() -> None:
    decision = resolve_mcp_runtime_mode(
        mcp_enabled=True,
        llm_provider="nvidia",
        nvidia_api_key=None,
    )

    assert decision.enabled is False
    assert decision.reason == "missing_nvidia_api_key"


def testinstall_mcp_generate_reply_guard_blocks_manual_generate_reply() -> None:
    session = _FakeSession()

    install_mcp_generate_reply_guard(session, mcp_runtime_active=True)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match=re.escape(MCP_GENERATE_REPLY_BLOCK_MESSAGE)):
        session.generate_reply(instructions="hello")


def testinstall_mcp_generate_reply_guard_is_noop_when_mcp_disabled() -> None:
    session = _FakeSession()

    install_mcp_generate_reply_guard(session, mcp_runtime_active=False)  # type: ignore[arg-type]
    session.generate_reply(instructions="hello")

    assert session.generate_reply_calls == [{"instructions": "hello"}]


def testrun_startup_greeting_uses_say_in_mcp_mode() -> None:
    session = _FakeSession()

    handle = run_startup_greeting(session, mcp_runtime_active=True)  # type: ignore[arg-type]

    assert handle is session.greeting_handle
    assert session.say_calls == [
        {
            "text": MCP_STARTUP_GREETING,
            "kwargs": {
                "allow_interruptions": True,
                "add_to_chat_ctx": True,
            },
        }
    ]
    assert session.generate_reply_calls == []


def testrun_startup_greeting_uses_generate_reply_without_mcp() -> None:
    session = _FakeSession()

    handle = run_startup_greeting(session, mcp_runtime_active=False)  # type: ignore[arg-type]

    assert handle is None
    assert session.generate_reply_calls == [
        {"instructions": "Greet the user and offer your assistance."}
    ]
    assert session.say_calls == []


def test_startup_greeting_monitor_times_out_and_interrupts() -> None:
    session = _FakeSession(greeting_handle=_FakeSpeechHandle(block=True))
    handle = run_startup_greeting(session, mcp_runtime_active=True)  # type: ignore[arg-type]

    assert handle is not None
    asyncio.run(monitor_startup_greeting_handle(handle, timeout_sec=0.01))

    assert handle.interrupt_calls == [True]


def testrun_startup_greeting_swallows_say_exception() -> None:
    session = _FailingSaySession()

    handle = run_startup_greeting(session, mcp_runtime_active=True)  # type: ignore[arg-type]

    assert handle is None
    assert session.say_calls == [{"text": MCP_STARTUP_GREETING, "kwargs": {"allow_interruptions": True, "add_to_chat_ctx": True}}]
    assert session.generate_reply_calls == []


def test_startup_greeting_monitor_handles_cancellation() -> None:
    async def _run() -> _FakeSpeechHandle:
        handle = _FakeSpeechHandle(block=True)
        task = asyncio.create_task(
            monitor_startup_greeting_handle(handle, timeout_sec=10.0)
        )
        await asyncio.sleep(0)
        task.cancel()
        await task
        return handle

    handle = asyncio.run(_run())
    assert handle.interrupt_calls == [True]


def testschedule_startup_greeting_task_is_shutdown_safe() -> None:
    async def _run() -> tuple[asyncio.Task[Any], _FakeSpeechHandle]:
        handle = _FakeSpeechHandle(block=True)
        session = _FakeSession(greeting_handle=handle)
        task = schedule_startup_greeting_task(  # type: ignore[arg-type]
            session,
            mcp_runtime_active=True,
        )
        assert task is not None
        assert not task.done()
        await cancel_task_for_shutdown(task, task_name="startup greeting", timeout_sec=0.1)
        return task, handle

    task, handle = asyncio.run(_run())
    assert task.done()
    assert handle.interrupt_calls == [True]


def testbuild_mcp_http_timeout_uses_runtime_timeout_for_all_phases() -> None:
    timeout = build_mcp_http_timeout(12.0)

    assert timeout.connect == 12.0
    assert timeout.read == 12.0
    assert timeout.write == 12.0
    assert timeout.pool == 12.0


def testbuild_mcp_http_timeout_enforces_positive_minimum() -> None:
    timeout = build_mcp_http_timeout(0.0)

    assert timeout.connect == 1.0
    assert timeout.read == 1.0
    assert timeout.write == 1.0
    assert timeout.pool == 1.0


def test_build_llm_runtime_supports_ollama_with_mcp() -> None:
    runtime = build_llm_runtime(
        llm_provider="ollama",
        llm_temperature=0.7,
        llm_max_tokens=1024,
        llm_timeout_sec=12.0,
        nvidia_api_key=None,
        nvidia_model="meta/llama-3.1-8b-instruct",
        ollama_base_url="http://localhost:11434/v1",
        ollama_model="qwen2.5:7b",
        ollama_api_key=None,
        mcp_enabled=True,
        mcp_server_url="https://huggingface.co/mcp",
    )

    assert runtime.provider == "ollama"
    assert runtime.model == "qwen2.5:7b"
    assert runtime.mcp_runtime_active is True
    assert runtime.mcp_servers is not None


def test_build_llm_runtime_requires_nvidia_key() -> None:
    with pytest.raises(ValueError, match="NVIDIA_API_KEY is required"):
        build_llm_runtime(
            llm_provider="nvidia",
            llm_temperature=0.7,
            llm_max_tokens=1024,
            llm_timeout_sec=12.0,
            nvidia_api_key=None,
            nvidia_model="meta/llama-3.1-8b-instruct",
            ollama_base_url="http://localhost:11434/v1",
            ollama_model="qwen2.5:7b",
            ollama_api_key="ollama",
            mcp_enabled=False,
            mcp_server_url="https://huggingface.co/mcp",
        )


def test_assistant_instructions_discourage_tools_for_small_talk() -> None:
    assert "greetings, acknowledgements, thanks, and casual small talk" in ASSISTANT_INSTRUCTIONS
    assert "do not call tools" in ASSISTANT_INSTRUCTIONS


def test_assistant_instructions_restrict_tool_usage_to_clear_intent() -> None:
    assert "only when user intent clearly requires external or up-to-date information" in ASSISTANT_INSTRUCTIONS
    assert "If a request can be answered directly from context and general knowledge, do not call tools." in ASSISTANT_INSTRUCTIONS


def test_assistant_instructions_enforce_ultra_short_answers() -> None:
    assert "answer with the fewest words possible" in ASSISTANT_INSTRUCTIONS
    assert "Keep most responses to one short sentence." in ASSISTANT_INSTRUCTIONS


def testrun_llm_warmup_consumes_first_chunk_and_closes_stream() -> None:
    stream = _FakeLLMStream(chunks=[{"delta": "OK"}, {"delta": "ignored"}])
    llm_client = _FakeLLMClient(stream)

    asyncio.run(
        run_llm_warmup(
            llm_client=llm_client,
            conn_options=object(),  # type: ignore[arg-type]
            provider="nvidia",
            model="test-model",
        )
    )

    assert len(llm_client.chat_calls) == 1
    call = llm_client.chat_calls[0]
    assert "chat_ctx" in call
    assert call["tools"] is None
    assert stream.iterations == 1
    assert stream.closed is True
    assert stream.aclose_calls == 1


def testrun_llm_warmup_swallows_stream_errors_and_closes() -> None:
    stream = _FakeLLMStream(iter_error=RuntimeError("warmup boom"))
    llm_client = _FakeLLMClient(stream)

    asyncio.run(
        run_llm_warmup(
            llm_client=llm_client,
            conn_options=object(),  # type: ignore[arg-type]
            provider="nvidia",
            model="test-model",
        )
    )

    assert len(llm_client.chat_calls) == 1
    assert stream.closed is True
    assert stream.aclose_calls == 1


def testschedule_llm_warmup_task_is_shutdown_safe() -> None:
    async def _run() -> tuple[asyncio.Task[Any], _FakeLLMStream]:
        stream = _FakeLLMStream(block_forever=True)
        llm_client = _FakeLLMClient(stream)
        task = schedule_llm_warmup_task(
            llm_client=llm_client,
            conn_options=object(),  # type: ignore[arg-type]
            provider="nvidia",
            model="test-model",
        )
        assert not task.done()
        await asyncio.sleep(0)
        await cancel_task_for_shutdown(task, task_name="llm warm-up", timeout_sec=0.1)
        return task, stream

    task, stream = asyncio.run(_run())
    assert task.done()
    assert stream.closed is True
    assert stream.aclose_calls == 1


def testinject_pre_tool_feedback_uses_model_leadin_before_tool_call() -> None:
    feedback = _FakeToolFeedback()
    input_chunk = _tool_chunk(content="I'll check that now.")

    output = asyncio.run(
        _collect(
            inject_pre_tool_feedback(
                _iter_items([input_chunk]),
                tool_feedback=feedback,  # type: ignore[arg-type]
            )
        )
    )

    assert output[0] == "I'll check that now."
    assert isinstance(output[1], llm.ChatChunk)
    assert output[1].delta is not None
    assert output[1].delta.content is None
    assert len(output[1].delta.tool_calls) == 1
    assert feedback.start_typing_calls == 1


def testinject_pre_tool_feedback_uses_fallback_when_model_omits_leadin() -> None:
    feedback = _FakeToolFeedback(fallback_phrase="One sec, checking now.")
    input_chunk = _tool_chunk(content=None)

    output = asyncio.run(
        _collect(
            inject_pre_tool_feedback(
                _iter_items([input_chunk]),
                tool_feedback=feedback,  # type: ignore[arg-type]
            )
        )
    )

    assert output[0] == "One sec, checking now."
    assert output[1] == input_chunk
    assert feedback.start_typing_calls == 1


def testinject_pre_tool_feedback_announces_once_per_tool_step() -> None:
    feedback = _FakeToolFeedback(fallback_phrase="Checking that for you.")
    first = _tool_chunk(content=None, call_id="call-1")
    second = _tool_chunk(content=None, call_id="call-2")

    output = asyncio.run(
        _collect(
            inject_pre_tool_feedback(
                _iter_items([first, second]),
                tool_feedback=feedback,  # type: ignore[arg-type]
            )
        )
    )

    assert output[0] == "Checking that for you."
    assert output[1] == first
    assert output[2] == second
    assert feedback.start_typing_calls == 1


def testinject_pre_tool_feedback_marks_tool_step_once() -> None:
    callback = _FakeToolStepStartedCallback()
    first = _tool_chunk(content=None, call_id="call-1")
    second = _tool_chunk(content=None, call_id="call-2")

    _ = asyncio.run(
        _collect(
            inject_pre_tool_feedback(
                _iter_items([first, second]),
                tool_feedback=None,
                on_tool_step_started=callback,
            )
        )
    )

    assert callback.calls == 1


def testinject_pre_tool_feedback_does_not_modify_non_tool_chunks() -> None:
    feedback = _FakeToolFeedback()
    non_tool_chunk = llm.ChatChunk(
        id="chunk-plain",
        delta=llm.ChoiceDelta(content="Hello there."),
    )

    output = asyncio.run(
        _collect(
            inject_pre_tool_feedback(
                _iter_items([non_tool_chunk]),
                tool_feedback=feedback,  # type: ignore[arg-type]
            )
        )
    )

    assert output == [non_tool_chunk]
    assert feedback.start_typing_calls == 0
