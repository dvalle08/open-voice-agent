"""LLM stream pre-tool feedback injection."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable
from typing import Any

from livekit.agents import llm

from src.agent.prompts.runtime import TOOL_PRE_SPEECH_FALLBACK
from src.agent.tools.feedback import ToolFeedbackController
from src.core.logger import logger


async def inject_pre_tool_feedback(
    source: AsyncIterable[Any],
    *,
    tool_feedback: ToolFeedbackController | None,
    on_tool_step_started: Callable[[], Awaitable[bool | None]] | None = None,
    should_announce_tool_step: Callable[[], Awaitable[bool]] | None = None,
    allowed_tool_names: set[str] | None = None,
) -> AsyncGenerator[Any, None]:
    tool_step_started = False

    async for chunk in source:
        if not isinstance(chunk, llm.ChatChunk):
            yield chunk
            continue

        delta = chunk.delta
        has_tool_calls = bool(delta and delta.tool_calls)
        if not has_tool_calls:
            yield chunk
            continue

        tool_call_names = _extract_tool_call_names(delta)
        if not _tool_calls_supported(tool_call_names, allowed_tool_names):
            logger.info(
                "tool_pre_speech_skipped reason=unknown_tool_names names=%s",
                ",".join(sorted(tool_call_names)) if tool_call_names else "<none>",
            )
            yield chunk
            continue

        if not tool_step_started:
            tool_step_started = True
            should_announce = True
            if should_announce_tool_step is not None:
                try:
                    should_announce = bool(await should_announce_tool_step())
                except Exception as exc:
                    logger.debug("should_announce_tool_step callback failed: %s", exc)
            elif on_tool_step_started is not None:
                try:
                    await on_tool_step_started()
                except Exception as exc:
                    logger.debug("tool_step_started callback failed: %s", exc)

            if not should_announce:
                logger.debug("tool_pre_speech_skipped reason=announcement_suppressed")
                yield chunk
                continue

            leadin_text = (delta.content or "").strip() if delta is not None else ""
            if leadin_text:
                logger.info(
                    "tool_pre_speech_source=model tool_pre_speech_text=%s",
                    leadin_text,
                )
                yield leadin_text

                rewritten = chunk.model_copy(deep=True)
                if rewritten.delta is not None:
                    rewritten.delta.content = None
                if tool_feedback is not None:
                    await tool_feedback.start_typing_sound()
                yield rewritten
                continue

            if tool_feedback is not None:
                fallback = tool_feedback.next_fallback_phrase()
            else:
                fallback = TOOL_PRE_SPEECH_FALLBACK
            logger.info(
                "tool_pre_speech_source=fallback tool_pre_speech_text=%s",
                fallback,
            )
            yield fallback
            if tool_feedback is not None:
                await tool_feedback.start_typing_sound()
            yield chunk
            continue

        yield chunk


def _extract_tool_call_names(delta: Any) -> set[str]:
    tool_call_names: set[str] = set()
    tool_calls = getattr(delta, "tool_calls", None) or []
    for tool_call in tool_calls:
        name = _normalize_tool_name(getattr(tool_call, "name", None))
        if name:
            tool_call_names.add(name)
            continue
        function = getattr(tool_call, "function", None)
        function_name = _normalize_tool_name(getattr(function, "name", None))
        if function_name:
            tool_call_names.add(function_name)
    return tool_call_names


def _normalize_tool_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _tool_calls_supported(
    tool_call_names: set[str],
    allowed_tool_names: set[str] | None,
) -> bool:
    if allowed_tool_names is None:
        return True
    if not tool_call_names:
        return True
    return any(name in allowed_tool_names for name in tool_call_names)
