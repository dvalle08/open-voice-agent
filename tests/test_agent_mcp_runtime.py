from __future__ import annotations

import asyncio
import re
from datetime import date
from typing import Any

import pytest
from livekit.agents import llm

from src.agent.models.llm_runtime import (
    MCP_GENERATE_REPLY_BLOCK_MESSAGE,
    build_llm_runtime,
    build_mcp_http_timeout,
    install_mcp_generate_reply_guard,
    resolve_mcp_server_urls,
    resolve_mcp_runtime_mode,
    run_startup_greeting,
)
from src.agent.prompts.assistant import (
    ASSISTANT_INSTRUCTIONS,
    build_assistant_instructions,
)
from src.agent.prompts.runtime import MCP_STARTUP_GREETING
from src.agent.runtime.tasks import (
    cancel_task_for_shutdown,
    monitor_startup_greeting_handle,
    run_llm_warmup,
    schedule_llm_warmup_task,
    schedule_startup_greeting_task,
)
from src.agent.tools.pre_tool_feedback import inject_pre_tool_feedback
from src.core.settings import LLMSettings, Settings


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


def _build_llm_settings(**overrides: Any) -> LLMSettings:
    defaults: dict[str, Any] = {
        "LLM_PROVIDER": "ollama",
        "OLLAMA_CLOUD_MODE": False,
        "OLLAMA_MODEL": "qwen2.5:7b",
        "OLLAMA_API_KEY": "ollama",
        "MCP_ENABLED": False,
        "MCP_SERVER_URL": "https://huggingface.co/mcp",
        "MCP_EXTRA_SERVER_URLS": "https://docs.livekit.io/mcp",
        "LLM_TEMPERATURE": 0.7,
        "LLM_MAX_TOKENS": 1024,
        "LLM_CONN_TIMEOUT_SEC": 12.0,
        "NVIDIA_MODEL": "meta/llama-3.1-8b-instruct",
        "NVIDIA_API_KEY": None,
    }
    defaults.update(overrides)
    return LLMSettings(**defaults)


class _FakeToolFeedback:
    def __init__(self, *, fallback_phrase: str = "Let me check that.") -> None:
        self.fallback_phrase = fallback_phrase
        self.start_typing_calls = 0

    def next_fallback_phrase(self) -> str:
        return self.fallback_phrase

    async def start_typing_sound(self) -> None:
        self.start_typing_calls += 1


class _FakeToolStepStartedCallback:
    def __init__(self, *, announce: bool = True) -> None:
        self.calls = 0
        self.announce = announce

    async def __call__(self) -> bool:
        self.calls += 1
        return self.announce


async def _iter_items(items: list[Any]) -> Any:
    for item in items:
        yield item


async def _collect(items: Any) -> list[Any]:
    out: list[Any] = []
    async for item in items:
        out.append(item)
    return out


def _tool_chunk(
    *,
    content: str | None,
    call_id: str = "call-1",
    name: str = "paper_search",
) -> llm.ChatChunk:
    return llm.ChatChunk(
        id="chunk-1",
        delta=llm.ChoiceDelta(
            content=content,
            tool_calls=[
                llm.FunctionToolCall(
                    name=name,
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
                "add_to_chat_ctx": False,
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


def test_startup_greeting_monitor_allows_unbounded_playout_when_timeout_disabled() -> None:
    async def _run() -> _FakeSpeechHandle:
        handle = _FakeSpeechHandle(block=True)
        task = asyncio.create_task(
            monitor_startup_greeting_handle(handle, timeout_sec=0.0)
        )
        await asyncio.sleep(0)
        assert not task.done()
        handle.interrupt(force=False)
        await task
        return handle

    handle = asyncio.run(_run())
    assert handle.interrupt_calls == [False]


def testrun_startup_greeting_swallows_say_exception() -> None:
    session = _FailingSaySession()

    handle = run_startup_greeting(session, mcp_runtime_active=True)  # type: ignore[arg-type]

    assert handle is None
    assert session.say_calls == [{"text": MCP_STARTUP_GREETING, "kwargs": {"allow_interruptions": True, "add_to_chat_ctx": False}}]
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
        _build_llm_settings(
            MCP_ENABLED=True,
            OLLAMA_API_KEY=None,
        )
    )

    assert runtime.provider == "ollama"
    assert runtime.model == "qwen2.5:7b"
    assert runtime.mcp_runtime_active is True
    assert runtime.mcp_servers is not None
    assert [server.url for server in runtime.mcp_servers] == [
        "https://huggingface.co/mcp",
        "https://docs.livekit.io/mcp",
    ]


def test_resolve_mcp_server_urls_trims_ignores_empty_and_deduplicates() -> None:
    urls = resolve_mcp_server_urls(
        mcp_server_url=" https://huggingface.co/mcp ",
        mcp_extra_server_urls=(
            "https://docs.livekit.io/mcp, , https://huggingface.co/mcp, "
            "https://docs.livekit.io/mcp, https://example.com/mcp"
        ),
    )

    assert urls == [
        "https://huggingface.co/mcp",
        "https://docs.livekit.io/mcp",
        "https://example.com/mcp",
    ]


def test_build_llm_runtime_rejects_cloud_alias_model_for_ollama_cloud_v1() -> None:
    with pytest.raises(
        ValueError,
        match=r"cannot use ':cloud' aliases.*https://ollama\.com/v1",
    ):
        build_llm_runtime(
            _build_llm_settings(
                OLLAMA_CLOUD_MODE=True,
                OLLAMA_MODEL="qwen3.5:cloud",
                OLLAMA_API_KEY="test-key",
                MCP_ENABLED=True,
            )
        )


def test_build_llm_runtime_requires_nvidia_key() -> None:
    with pytest.raises(ValueError, match="NVIDIA_API_KEY is required"):
        build_llm_runtime(
            _build_llm_settings(
                LLM_PROVIDER="nvidia",
                NVIDIA_API_KEY=None,
            )
        )


def test_build_llm_runtime_passes_nvidia_disable_thinking_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}
    fake_llm = object()

    def _fake_openai_llm(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return fake_llm

    monkeypatch.setattr(
        "src.agent.models.llm_runtime.openai_plugin.LLM",
        _fake_openai_llm,
    )

    runtime = build_llm_runtime(
        _build_llm_settings(
            LLM_PROVIDER="nvidia",
            NVIDIA_API_KEY="nvapi-test",
            NVIDIA_MODEL="qwen/qwen3-next-80b-a3b-instruct",
        )
    )

    assert runtime.llm is fake_llm
    assert captured_kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert captured_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_build_llm_runtime_passes_ollama_disable_thinking_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}
    fake_llm = object()

    def _fake_openai_llm(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return fake_llm

    monkeypatch.setattr(
        "src.agent.models.llm_runtime.openai_plugin.LLM",
        _fake_openai_llm,
    )

    runtime = build_llm_runtime(_build_llm_settings())

    assert runtime.llm is fake_llm
    assert captured_kwargs["base_url"] == "http://localhost:11434/v1"
    assert captured_kwargs["extra_body"] == {"think": False}


def test_assistant_instructions_enforce_ultra_short_answers() -> None:
    assert "Answer as quickly as possible." in ASSISTANT_INSTRUCTIONS
    assert "Give the shortest complete answer." in ASSISTANT_INSTRUCTIONS
    assert "Default to one short sentence." in ASSISTANT_INSTRUCTIONS
    assert "If a few words are enough, use a few words." in ASSISTANT_INSTRUCTIONS
    assert "Expand only when the user asks or accuracy requires it." in ASSISTANT_INSTRUCTIONS
    assert "The user only speaks to you, not types, so do not ask them to write, type, paste, or use chat." in ASSISTANT_INSTRUCTIONS


def test_assistant_instructions_prioritize_knowledge_before_tools() -> None:
    assert "Answer from your own knowledge first, then from the setup summary." in ASSISTANT_INSTRUCTIONS


def test_assistant_instructions_limit_huggingface_and_livekit_tools() -> None:
    assert "Do not claim you can generate or edit images." in ASSISTANT_INSTRUCTIONS
    assert "Use Hugging Face tools only when needed and only if they are available in the current session." in ASSISTANT_INSTRUCTIONS
    assert "Use LiveKit tools only for LiveKit-specific questions not covered by your own knowledge or the setup summary." in ASSISTANT_INSTRUCTIONS


def test_assistant_instructions_describe_pipeline_identity() -> None:
    assert (
        "You are Open Voice Agent, a real-time voice pipeline."
        in ASSISTANT_INSTRUCTIONS
    )
    assert (
        "You are a pipeline, not a single model."
        in ASSISTANT_INSTRUCTIONS
    )
    assert (
        "For self-description, describe the pipeline briefly and mention VAD, turn detection, STT, LLM, and TTS only when relevant."
        in ASSISTANT_INSTRUCTIONS
    )


def test_assistant_instructions_handle_setup_questions_from_summary() -> None:
    assert "For setup questions, answer from the setup summary." in ASSISTANT_INSTRUCTIONS


def test_build_assistant_instructions_includes_current_date() -> None:
    instructions = build_assistant_instructions(current_date=date(2026, 3, 7))
    assert "Current date: March 7, 2026." in instructions


def test_assistant_instructions_include_capabilities_and_limitations() -> None:
    instructions = build_assistant_instructions(current_date=date(2026, 3, 7))
    assert "Priority 1. Identity." in instructions
    assert "Priority 2. Response style." in instructions
    assert "Priority 3. Tools." in instructions
    assert "Priority 4. Safety and setup." in instructions
    assert "The user only speaks to you, not types" in instructions
    assert "Do not claim you can generate or edit images." in instructions
    assert "Only call tools that are available in the current session." in instructions
    assert "Never invent tools." in instructions
    assert "Use plain voice-friendly text only" in instructions


def test_build_assistant_instructions_include_configuration_summary() -> None:
    instructions = build_assistant_instructions(current_date=date(2026, 3, 7))

    assert "Setup summary:" in instructions
    assert "Voice stack: STT provider=" in instructions
    assert "LLM provider=" in instructions
    assert "TTS provider=" in instructions
    assert "LiveKit: runtime agent_name=" in instructions
    assert "audio sample_rate=" in instructions
    assert "MCP runtime:" in instructions


def test_build_assistant_instructions_hide_pocket_specific_details_for_deepgram() -> None:
    settings = Settings()
    settings.voice.TTS_PROVIDER = "deepgram"
    settings.voice.DEEPGRAM_API_KEY = "deepgram-secret-test-value"

    instructions = build_assistant_instructions(
        current_date=date(2026, 3, 7),
        current_settings=settings,
    )

    assert "TTS provider=deepgram, model=plugin-default" in instructions
    assert "voice=alba" not in instructions
    assert "lsd_decode_steps=" not in instructions


def test_build_assistant_instructions_hide_pocket_specific_details_for_nvidia_tts() -> None:
    settings = Settings()
    settings.voice.TTS_PROVIDER = "nvidia"
    settings.voice.NVIDIA_TTS_VOICE = "Magpie-Multilingual.EN-US.Leo"
    settings.voice.NVIDIA_TTS_LANGUAGE_CODE = "en-US"
    settings.voice.NVIDIA_TTS_SERVER = "grpc.nvcf.nvidia.com:443"
    settings.voice.NVIDIA_TTS_USE_SSL = True
    settings.llm.NVIDIA_API_KEY = "nvapi-secret-test-value"

    instructions = build_assistant_instructions(
        current_date=date(2026, 3, 7),
        current_settings=settings,
    )

    assert (
        "TTS provider=nvidia, voice=Magpie-Multilingual.EN-US.Leo, "
        "language_code=en-US, server=grpc.nvcf.nvidia.com:443, "
        "use_ssl=True, timeout_sec=45.0."
    ) in instructions
    assert "language_code=en-US" in instructions
    assert "server=grpc.nvcf.nvidia.com:443" in instructions
    assert "use_ssl=True" in instructions
    assert "voice=alba" not in instructions
    assert "lsd_decode_steps=" not in instructions


def test_build_assistant_instructions_include_deepgram_stt_details() -> None:
    settings = Settings()
    settings.voice.DEEPGRAM_API_KEY = "deepgram-secret-test-value"
    settings.stt.STT_PROVIDER = "deepgram"
    settings.stt.DEEPGRAM_STT_MODEL = "nova-3"
    settings.stt.DEEPGRAM_STT_LANGUAGE = "en-US"

    instructions = build_assistant_instructions(
        current_date=date(2026, 3, 7),
        current_settings=settings,
    )

    assert "STT provider=deepgram, model=nova-3, language=en-US;" in instructions
    assert "usefulsensors/moonshine-streaming-medium" not in instructions
    assert "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer" not in instructions


def test_build_assistant_instructions_redacts_sensitive_values() -> None:
    settings = Settings()
    settings.voice.DEEPGRAM_API_KEY = "deepgram-secret-test-value"
    settings.voice.NVIDIA_TTS_API_KEY = "nvidia-tts-secret-test-value"
    settings.llm.NVIDIA_API_KEY = "nvapi-secret-test-value"
    settings.stt.NVIDIA_STT_API_KEY = "nvidia-stt-secret-test-value"
    settings.livekit.LIVEKIT_API_KEY = "livekit-api-key-test-value"
    settings.livekit.LIVEKIT_API_SECRET = "livekit-api-secret-test-value"

    instructions = build_assistant_instructions(
        current_date=date(2026, 3, 7),
        current_settings=settings,
    )

    assert "deepgram-secret-test-value" not in instructions
    assert "nvidia-tts-secret-test-value" not in instructions
    assert "nvapi-secret-test-value" not in instructions
    assert "nvidia-stt-secret-test-value" not in instructions
    assert "livekit-api-key-test-value" not in instructions
    assert "livekit-api-secret-test-value" not in instructions
    assert "Credential state (redacted):" in instructions
    assert "<redacted>" in instructions


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


def testinject_pre_tool_feedback_skips_unknown_tool_names() -> None:
    feedback = _FakeToolFeedback()
    unknown_tool_chunk = _tool_chunk(content=None, name="explain_pipeline")

    output = asyncio.run(
        _collect(
            inject_pre_tool_feedback(
                _iter_items([unknown_tool_chunk]),
                tool_feedback=feedback,  # type: ignore[arg-type]
                allowed_tool_names={"paper_search"},
            )
        )
    )

    assert output == [unknown_tool_chunk]
    assert feedback.start_typing_calls == 0


def testinject_pre_tool_feedback_respects_announcement_suppression() -> None:
    feedback = _FakeToolFeedback()
    callback = _FakeToolStepStartedCallback(announce=False)
    tool_chunk = _tool_chunk(content=None)

    output = asyncio.run(
        _collect(
            inject_pre_tool_feedback(
                _iter_items([tool_chunk]),
                tool_feedback=feedback,  # type: ignore[arg-type]
                should_announce_tool_step=callback,
                allowed_tool_names={"paper_search"},
            )
        )
    )

    assert output == [tool_chunk]
    assert callback.calls == 1
    assert feedback.start_typing_calls == 0


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
