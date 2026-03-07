from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from livekit.agents import AgentSession, mcp
from livekit.plugins import openai as openai_plugin

from src.agent.prompts.runtime import MCP_STARTUP_GREETING
from src.core.logger import logger
from src.core.settings import LLMSettings

NVIDIA_OPENAI_BASE_URL = "https://integrate.api.nvidia.com/v1"
MCP_STARTUP_GREETING_TIMEOUT_SEC = 8.0
MCP_GENERATE_REPLY_BLOCK_MESSAGE = (
    "Manual generate_reply is disabled in MCP mode; use session.say(...) instead."
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


def build_llm_runtime(
    llm_settings: LLMSettings,
) -> LLMRuntimeConfig:
    provider = (llm_settings.LLM_PROVIDER or "").strip().lower()
    timeout = build_mcp_http_timeout(llm_settings.LLM_CONN_TIMEOUT_SEC)
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
        mcp_servers = [mcp.MCPServerHTTP(url=url) for url in mcp_server_urls]
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
            "MCP runtime enabled: mcp_servers=%s llm_provider=%s llm_model=%s llm_timeout_sec=%.2f",
            mcp_server_urls,
            provider,
            model,
            llm_settings.LLM_CONN_TIMEOUT_SEC,
        )
    else:
        logger.info("MCP runtime disabled (MCP_ENABLED=false)")

    llm = openai_plugin.LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=llm_settings.LLM_TEMPERATURE,
        max_completion_tokens=llm_settings.LLM_MAX_TOKENS,
        timeout=timeout,
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
