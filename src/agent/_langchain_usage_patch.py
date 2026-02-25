"""Patch LiveKit LangChain bridge to propagate token usage into LLMMetrics.

LiveKit computes LLMMetrics token fields from ``ChatChunk.usage``. The upstream
``livekit.plugins.langchain.langgraph`` bridge emits chunks with content but, in
some providers, does not map LangChain ``usage_metadata`` into ``usage``.
When that happens, Langfuse receives token metrics as zeros.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from langchain_core.messages import BaseMessageChunk
from livekit.agents import llm, utils
from livekit.plugins.langchain import langgraph as lk_langgraph

from src.core.logger import logger

_PATCH_FLAG = "_open_voice_agent_usage_patch_applied"
_ORIGINAL_TO_CHAT_CHUNK = "_open_voice_agent_original_to_chat_chunk"
_MISSING_USAGE_LOGGED = "_open_voice_agent_missing_usage_logged"


def apply_langchain_usage_patch() -> bool:
    """Apply an idempotent patch to preserve token usage from LangChain chunks."""
    if getattr(lk_langgraph, _PATCH_FLAG, False):
        return False

    original = getattr(lk_langgraph, "_to_chat_chunk", None)
    if not callable(original):
        logger.warning("LangChain usage patch skipped: _to_chat_chunk not found")
        return False

    def _patched_to_chat_chunk(msg: str | Any) -> llm.ChatChunk | None:
        message_id = utils.shortuuid("LC_")
        content: str | None = None
        usage = _completion_usage_from_message_chunk(msg)

        if isinstance(msg, str):
            content = msg
        elif isinstance(msg, BaseMessageChunk):
            text_value = getattr(msg, "text", None)
            if text_value is not None:
                content = str(text_value)
            chunk_id = getattr(msg, "id", None)
            if chunk_id:
                message_id = chunk_id  # type: ignore[assignment]

        # Preserve usage-only chunks (final event often carries usage with no text).
        if not content and usage is None:
            return None

        delta = (
            llm.ChoiceDelta(
                role="assistant",
                content=content,
            )
            if content
            else None
        )
        return llm.ChatChunk(
            id=message_id,
            delta=delta,
            usage=usage,
        )

    setattr(lk_langgraph, _ORIGINAL_TO_CHAT_CHUNK, original)
    setattr(lk_langgraph, "_to_chat_chunk", _patched_to_chat_chunk)
    setattr(lk_langgraph, _PATCH_FLAG, True)
    logger.info("Applied LangChain usage bridge patch for LLM token metrics")
    return True


def _completion_usage_from_message_chunk(
    message_chunk: Any,
) -> Optional[llm.CompletionUsage]:
    if not isinstance(message_chunk, BaseMessageChunk):
        return None

    usage_metadata = _as_mapping(getattr(message_chunk, "usage_metadata", None))
    if usage_metadata:
        usage = _completion_usage_from_mapping(usage_metadata)
        if usage:
            return usage

    # Fallback for providers that place usage in response_metadata.
    response_metadata = _as_mapping(getattr(message_chunk, "response_metadata", None))
    if not response_metadata:
        _log_missing_usage_once(message_chunk)
        return None

    response_usage = (
        _as_mapping(response_metadata.get("usage_metadata"))
        or _as_mapping(response_metadata.get("usage"))
        or _as_mapping(response_metadata.get("token_usage"))
    )
    usage = _completion_usage_from_mapping(response_usage) if response_usage else None
    if usage is None:
        _log_missing_usage_once(message_chunk)
    return usage


def _completion_usage_from_mapping(
    usage: Mapping[str, Any] | None,
) -> Optional[llm.CompletionUsage]:
    if not usage:
        return None

    prompt_tokens = _as_int(
        usage.get("input_tokens"),
        usage.get("prompt_tokens"),
    )
    completion_tokens = _as_int(
        usage.get("output_tokens"),
        usage.get("completion_tokens"),
    )
    total_tokens = _as_int(usage.get("total_tokens"))

    # Derive missing pieces when enough data is present.
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens is None and total_tokens is not None and completion_tokens is not None:
        prompt_tokens = max(total_tokens - completion_tokens, 0)
    if completion_tokens is None and total_tokens is not None and prompt_tokens is not None:
        completion_tokens = max(total_tokens - prompt_tokens, 0)

    if prompt_tokens is None or completion_tokens is None or total_tokens is None:
        return None

    prompt_cached_tokens = _extract_prompt_cached_tokens(usage)
    return llm.CompletionUsage(
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        prompt_cached_tokens=prompt_cached_tokens,
        total_tokens=total_tokens,
    )


def _extract_prompt_cached_tokens(usage: Mapping[str, Any]) -> int:
    cached = _as_int(
        usage.get("prompt_cached_tokens"),
        usage.get("cached_tokens"),
        usage.get("cache_read_tokens"),
    )
    if cached is not None:
        return max(cached, 0)

    input_details = _as_mapping(usage.get("input_token_details"))
    if input_details:
        cached = _as_int(
            input_details.get("cache_read"),
            input_details.get("cache_read_tokens"),
            input_details.get("cached_tokens"),
        )
        if cached is not None:
            return max(cached, 0)

    prompt_details = _as_mapping(usage.get("prompt_tokens_details"))
    if prompt_details:
        cached = _as_int(
            prompt_details.get("cached_tokens"),
            prompt_details.get("cache_read"),
            prompt_details.get("cache_read_tokens"),
        )
        if cached is not None:
            return max(cached, 0)

    return 0


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _as_int(*values: Any) -> Optional[int]:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
    return None


def _log_missing_usage_once(message_chunk: BaseMessageChunk) -> None:
    if getattr(lk_langgraph, _MISSING_USAGE_LOGGED, False):
        return
    setattr(lk_langgraph, _MISSING_USAGE_LOGGED, True)
    logger.info(
        "LLM chunk arrived without token usage metadata; LLM token metrics may remain zero. chunk_type=%s",
        type(message_chunk).__name__,
    )
