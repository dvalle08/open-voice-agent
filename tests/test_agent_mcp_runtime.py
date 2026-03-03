from __future__ import annotations

import asyncio
import re

import pytest

from src.agent.agent import ASSISTANT_INSTRUCTIONS
from src.agent.llm_runtime import (
    MCP_GENERATE_REPLY_BLOCK_MESSAGE,
    MCP_STARTUP_GREETING,
    _build_mcp_http_timeout,
    _install_mcp_generate_reply_guard,
    _run_startup_greeting,
    build_llm_runtime,
    resolve_mcp_runtime_mode,
)


class _FakeSession:
    def __init__(self) -> None:
        self.say_calls: list[dict[str, object]] = []
        self.generate_reply_calls: list[dict[str, object]] = []

    async def say(self, text: str, **kwargs: object) -> str:
        self.say_calls.append({"text": text, "kwargs": kwargs})
        return "say-handle"

    def generate_reply(self, **kwargs: object) -> str:
        self.generate_reply_calls.append(kwargs)
        return "reply-handle"


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


def test_install_mcp_generate_reply_guard_blocks_manual_generate_reply() -> None:
    session = _FakeSession()

    _install_mcp_generate_reply_guard(session, mcp_runtime_active=True)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match=re.escape(MCP_GENERATE_REPLY_BLOCK_MESSAGE)):
        session.generate_reply(instructions="hello")


def test_install_mcp_generate_reply_guard_is_noop_when_mcp_disabled() -> None:
    session = _FakeSession()

    _install_mcp_generate_reply_guard(session, mcp_runtime_active=False)  # type: ignore[arg-type]
    session.generate_reply(instructions="hello")

    assert session.generate_reply_calls == [{"instructions": "hello"}]


def test_run_startup_greeting_uses_say_in_mcp_mode() -> None:
    session = _FakeSession()

    asyncio.run(_run_startup_greeting(session, mcp_runtime_active=True))  # type: ignore[arg-type]

    assert session.say_calls == [{"text": MCP_STARTUP_GREETING, "kwargs": {}}]
    assert session.generate_reply_calls == []


def test_run_startup_greeting_uses_generate_reply_without_mcp() -> None:
    session = _FakeSession()

    asyncio.run(_run_startup_greeting(session, mcp_runtime_active=False))  # type: ignore[arg-type]

    assert session.generate_reply_calls == [
        {"instructions": "Greet the user and offer your assistance."}
    ]
    assert session.say_calls == []


def test_build_mcp_http_timeout_uses_runtime_timeout_for_all_phases() -> None:
    timeout = _build_mcp_http_timeout(12.0)

    assert timeout.connect == 12.0
    assert timeout.read == 12.0
    assert timeout.write == 12.0
    assert timeout.pool == 12.0


def test_build_mcp_http_timeout_enforces_positive_minimum() -> None:
    timeout = _build_mcp_http_timeout(0.0)

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
