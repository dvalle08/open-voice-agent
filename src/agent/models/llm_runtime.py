from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from mcp.shared.exceptions import McpError
from livekit.agents import AgentSession, mcp
from livekit.agents.llm.tool_context import ToolError
from livekit.plugins import openai as openai_plugin

from src.agent.prompts.runtime import MCP_STARTUP_GREETING
from src.core.logger import logger
from src.core.settings import LLMSettings

NVIDIA_OPENAI_BASE_URL = "https://integrate.api.nvidia.com/v1"
MCP_GENERATE_REPLY_BLOCK_MESSAGE = (
    "Manual generate_reply is disabled in MCP mode; use session.say(...) instead."
)
MCP_TOOL_TIMEOUT_MESSAGE = (
    "The external tool '{tool_name}' timed out. "
    "Do not retry '{tool_name}' again in this turn. "
    "Give the user a brief answer without it."
)
MCP_TOOL_UNAVAILABLE_MESSAGE = (
    "The external tool '{tool_name}' is temporarily unavailable. "
    "Do not retry '{tool_name}' again in this turn. "
    "Give the user a brief answer without it."
)


@dataclass(frozen=True)
class MCPRuntimeDecision:
    enabled: bool
    reason: str


@dataclass(frozen=True)
class LLMRuntimeConfig:
    llm: Any
    mcp_servers: list[mcp.MCPServerHTTP] | None
    provider: str
    model: str

    @property
    def mcp_runtime_active(self) -> bool:
        return self.mcp_servers is not None


class ConfiguredMCPServerHTTP(mcp.MCPServerHTTP):
    def __init__(
        self,
        *,
        url: str,
        timeout_seconds: float,
        headers: dict[str, Any] | None = None,
    ) -> None:
        bounded_timeout = _bounded_timeout_seconds(timeout_seconds)
        super().__init__(
            url=url,
            headers=headers,
            timeout=bounded_timeout,
            client_session_timeout_seconds=bounded_timeout,
        )
        self._request_timeout_seconds = bounded_timeout

    @property
    def request_timeout_seconds(self) -> float:
        return self._request_timeout_seconds

    @property
    def client_session_timeout_seconds(self) -> float:
        return self._read_timeout

    def _make_function_tool(
        self,
        name: str,
        description: str | None,
        input_schema: dict[str, Any],
        meta: dict[str, Any] | None,
    ) -> mcp.MCPTool:
        async def _tool_called(raw_arguments: dict[str, Any]) -> Any:
            if self._client is None:
                raise ToolError(
                    "Tool invocation failed: internal service is unavailable. "
                    "Please check that the MCPServer is still running."
                )

            try:
                tool_result = await self._client.call_tool(name, raw_arguments)
            except Exception as exc:
                normalized = normalize_mcp_tool_exception(tool_name=name, exc=exc)
                if normalized is None:
                    raise

                logger.warning(
                    "MCP tool invocation failed: tool=%s timeout=%s detail=%s",
                    name,
                    is_mcp_timeout_exception(exc),
                    describe_mcp_exception(exc),
                )
                raise normalized from exc

            if tool_result.isError:
                error_str = "\n".join(str(part) for part in tool_result.content)
                raise ToolError(error_str)

            if len(tool_result.content) == 1:
                return tool_result.content[0].model_dump_json()
            if len(tool_result.content) > 1:
                return json.dumps([item.model_dump() for item in tool_result.content])

            raise ToolError(
                f"Tool '{name}' completed without producing a result. "
                "This might indicate an issue with internal processing."
            )

        raw_schema = {
            "name": name,
            "description": description,
            "parameters": input_schema,
        }
        if meta:
            raw_schema["meta"] = meta

        return mcp.function_tool(_tool_called, raw_schema=raw_schema)


def resolve_mcp_runtime_mode(
    *,
    mcp_enabled: bool,
    llm_provider: str,
    nvidia_api_key: str | None,
) -> MCPRuntimeDecision:
    provider = (llm_provider or "").strip().lower()
    if not mcp_enabled:
        return MCPRuntimeDecision(enabled=False, reason="mcp_disabled")
    if provider not in {"nvidia", "ollama"}:
        return MCPRuntimeDecision(enabled=False, reason=f"provider_not_supported:{provider}")
    if provider == "nvidia" and not nvidia_api_key:
        return MCPRuntimeDecision(enabled=False, reason="missing_nvidia_api_key")
    return MCPRuntimeDecision(enabled=True, reason="mcp_enabled")


def resolve_mcp_server_urls(
    *,
    mcp_server_url: str,
    mcp_extra_server_urls: str,
) -> list[str]:
    candidates = [mcp_server_url, *(mcp_extra_server_urls or "").split(",")]
    deduplicated: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        normalized = (candidate or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(normalized)

    return deduplicated


def normalize_mcp_tool_exception(*, tool_name: str, exc: Exception) -> ToolError | None:
    if is_mcp_timeout_exception(exc):
        return ToolError(MCP_TOOL_TIMEOUT_MESSAGE.format(tool_name=tool_name))
    if is_mcp_transport_exception(exc):
        return ToolError(MCP_TOOL_UNAVAILABLE_MESSAGE.format(tool_name=tool_name))
    return None


def is_mcp_timeout_exception(exc: BaseException) -> bool:
    for error in iter_exception_chain(exc):
        if isinstance(error, (TimeoutError, httpx.TimeoutException)):
            return True
        if isinstance(error, McpError) and looks_like_timeout_message(error.error.message):
            return True
    return False


def is_mcp_transport_exception(exc: BaseException) -> bool:
    return any(
        isinstance(error, (McpError, httpx.RequestError, OSError))
        for error in iter_exception_chain(exc)
    )


def iter_exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__

    return tuple(chain)


def looks_like_timeout_message(message: str | None) -> bool:
    normalized = (message or "").strip().lower()
    if not normalized:
        return False
    return any(
        token in normalized
        for token in ("timed out", "timeout", "deadline exceeded", "read timed out")
    )


def describe_mcp_exception(exc: BaseException) -> str:
    for error in iter_exception_chain(exc):
        if isinstance(error, McpError):
            detail = error.error.message
        else:
            detail = str(error)
        if detail:
            return detail
    return type(exc).__name__


def _bounded_timeout_seconds(timeout_seconds: float) -> float:
    return max(float(timeout_seconds), 1.0)


def build_llm_runtime(
    llm_settings: LLMSettings,
) -> LLMRuntimeConfig:
    provider = (llm_settings.LLM_PROVIDER or "").strip().lower()
    llm_timeout = build_mcp_http_timeout(llm_settings.LLM_CONN_TIMEOUT_SEC)
    mcp_timeout_seconds = _bounded_timeout_seconds(llm_settings.MCP_CONN_TIMEOUT_SEC)
    mcp_decision = resolve_mcp_runtime_mode(
        mcp_enabled=llm_settings.MCP_ENABLED,
        llm_provider=provider,
        nvidia_api_key=llm_settings.NVIDIA_API_KEY,
    )
    mcp_server_urls: list[str] = []
    if mcp_decision.enabled:
        mcp_server_urls = resolve_mcp_server_urls(
            mcp_server_url=llm_settings.MCP_SERVER_URL,
            mcp_extra_server_urls=llm_settings.MCP_EXTRA_SERVER_URLS,
        )
        mcp_servers = [
            ConfiguredMCPServerHTTP(url=url, timeout_seconds=mcp_timeout_seconds)
            for url in mcp_server_urls
        ]
    else:
        mcp_servers = None

    if provider == "nvidia":
        if not llm_settings.NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA_API_KEY is required when LLM_PROVIDER=nvidia"
            )
        model = llm_settings.NVIDIA_MODEL
        base_url = NVIDIA_OPENAI_BASE_URL
        api_key = llm_settings.NVIDIA_API_KEY
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    elif provider == "ollama":
        model = (llm_settings.OLLAMA_MODEL or "").strip()
        if not model:
            raise ValueError("OLLAMA_MODEL is required when LLM_PROVIDER=ollama")
        base_url = (llm_settings.OLLAMA_BASE_URL or "").strip()
        if not base_url:
            raise ValueError("OLLAMA_BASE_URL is required when LLM_PROVIDER=ollama")
        validate_ollama_model_for_endpoint(base_url=base_url, model=model)
        api_key = resolve_ollama_api_key(llm_settings.OLLAMA_API_KEY)
        extra_body = {"think": False}
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Must be 'nvidia' or 'ollama'"
        )

    if llm_settings.MCP_ENABLED and not mcp_decision.enabled:
        logger.warning(
            "MCP runtime requested but unavailable: reason=%s provider=%s",
            mcp_decision.reason,
            provider,
        )
    elif mcp_decision.enabled:
        logger.info(
            "MCP runtime enabled: mcp_servers=%s llm_provider=%s llm_model=%s llm_timeout_sec=%.2f mcp_timeout_sec=%.2f",
            mcp_server_urls,
            provider,
            model,
            llm_settings.LLM_CONN_TIMEOUT_SEC,
            mcp_timeout_seconds,
        )
    else:
        logger.info("MCP runtime disabled (MCP_ENABLED=false)")

    llm = openai_plugin.LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=llm_settings.LLM_TEMPERATURE,
        max_completion_tokens=llm_settings.LLM_MAX_TOKENS,
        timeout=llm_timeout,
        _strict_tool_schema=False,
        extra_body=extra_body,
    )
    return LLMRuntimeConfig(
        llm=llm,
        mcp_servers=mcp_servers,
        provider=provider,
        model=model,
    )


def resolve_ollama_api_key(api_key: str | None) -> str:
    value = (api_key or "").strip()
    if value:
        return value
    return "ollama"


def validate_ollama_model_for_endpoint(*, base_url: str, model: str) -> None:
    if not is_ollama_cloud_openai_endpoint(base_url):
        return

    if model.lower().endswith(":cloud"):
        raise ValueError(
            "OLLAMA_MODEL cannot use ':cloud' aliases with OLLAMA_BASE_URL=https://ollama.com/v1. "
            "Use an exact model ID from https://ollama.com/v1/models (for example, qwen3-next:80b)."
        )


def is_ollama_cloud_openai_endpoint(base_url: str) -> bool:
    raw = (base_url or "").strip()
    if not raw:
        return False

    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").rstrip("/")
    return host in {"ollama.com", "www.ollama.com", "api.ollama.com"} and path == "/v1"


def build_mcp_http_timeout(timeout_seconds: float) -> httpx.Timeout:
    bounded_timeout = max(timeout_seconds, 1.0)
    return httpx.Timeout(
        connect=bounded_timeout,
        read=bounded_timeout,
        write=bounded_timeout,
        pool=bounded_timeout,
    )


def install_mcp_generate_reply_guard(
    session: AgentSession,
    *,
    mcp_runtime_active: bool,
) -> None:
    if not mcp_runtime_active:
        return
    if getattr(session, "_open_voice_mcp_generate_reply_guard_installed", False):
        return

    def _blocked_generate_reply(*_: Any, **__: Any) -> Any:
        raise RuntimeError(MCP_GENERATE_REPLY_BLOCK_MESSAGE)

    setattr(session, "_open_voice_mcp_generate_reply_guard_installed", True)
    setattr(session, "_open_voice_original_generate_reply", session.generate_reply)
    setattr(session, "generate_reply", _blocked_generate_reply)
    logger.info("MCP runtime policy active: manual generate_reply disabled")


def run_startup_greeting(
    session: AgentSession,
    *,
    mcp_runtime_active: bool,
) -> Any | None:
    if mcp_runtime_active:
        logger.info("MCP runtime startup greeting via session.say")
        try:
            return session.say(
                MCP_STARTUP_GREETING,
                allow_interruptions=True,
                add_to_chat_ctx=False,
            )
        except Exception as exc:
            logger.warning(f"MCP startup greeting could not start: {exc}")
            return None

    try:
        session.generate_reply(instructions="Greet the user and offer your assistance.")
    except Exception as exc:
        logger.warning(f"Startup greeting via generate_reply failed: {exc}")
    return None
