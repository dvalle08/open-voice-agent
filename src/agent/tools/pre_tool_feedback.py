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
    on_tool_step_started: Callable[[], Awaitable[None]] | None = None,
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

        if not tool_step_started:
            tool_step_started = True
            if on_tool_step_started is not None:
                try:
                    await on_tool_step_started()
                except Exception as exc:
                    logger.debug("tool_step_started callback failed: %s", exc)
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
