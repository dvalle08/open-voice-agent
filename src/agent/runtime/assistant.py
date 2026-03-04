"""Agent implementation with session event hooks."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from livekit.agents import Agent, llm
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
)

from src.agent.prompts.assistant import ASSISTANT_INSTRUCTIONS
from src.agent.tools.feedback import ToolFeedbackController
from src.agent.tools.pre_tool_feedback import inject_pre_tool_feedback
from src.agent.traces.errors import error_detail, error_recoverable, error_type_name
from src.agent.traces.metrics_collector import MetricsCollector
from src.core.logger import logger


class Assistant(Agent):
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        *,
        room_name: str,
        job_id: str,
        tool_feedback: ToolFeedbackController | None = None,
    ) -> None:
        super().__init__(
            instructions=ASSISTANT_INSTRUCTIONS,
        )
        self._metrics_collector = metrics_collector
        self._room_name = room_name
        self._job_id = job_id
        self._tool_feedback = tool_feedback

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool | llm.RawFunctionTool],
        model_settings: Any,
    ) -> AsyncGenerator[llm.ChatChunk | str | Any, None]:
        llm_node = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        if asyncio.iscoroutine(llm_node):
            llm_node = await llm_node

        if isinstance(llm_node, str):
            yield llm_node
            return

        if not isinstance(llm_node, AsyncIterable):
            return

        aclose = getattr(llm_node, "aclose", None)
        try:
            async for chunk in inject_pre_tool_feedback(
                llm_node,
                tool_feedback=self._tool_feedback,
                on_tool_step_started=self._metrics_collector.on_tool_step_started,
            ):
                yield chunk
        finally:
            if callable(aclose):
                with contextlib.suppress(Exception):
                    await aclose()

    async def on_enter(self) -> None:
        """Called when the agent enters the session. Set up metrics listeners."""

        def metrics_wrapper(event: MetricsCollectedEvent) -> None:
            asyncio.create_task(
                self._metrics_collector.on_metrics_collected(event.metrics)
            )

        def transcript_wrapper(event: UserInputTranscribedEvent) -> None:
            asyncio.create_task(
                self._metrics_collector.on_user_input_transcribed(
                    event.transcript,
                    is_final=event.is_final,
                )
            )

        def conversation_item_wrapper(event: ConversationItemAddedEvent) -> None:
            item = event.item
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            asyncio.create_task(
                self._metrics_collector.on_conversation_item_added(
                    role=role,
                    content=content,
                )
            )

        def speech_created_wrapper(event: SpeechCreatedEvent) -> None:
            asyncio.create_task(
                self._metrics_collector.on_speech_created(event.speech_handle)
            )

        def function_tools_executed_wrapper(event: FunctionToolsExecutedEvent) -> None:
            asyncio.create_task(
                self._metrics_collector.on_function_tools_executed(
                    function_calls=event.function_calls,
                    function_call_outputs=event.function_call_outputs,
                    created_at=event.created_at,
                )
            )
            if self._tool_feedback is not None:
                asyncio.create_task(
                    self._tool_feedback.stop_typing_sound(reason="function_tools_executed")
                )

        def agent_state_changed_wrapper(event: AgentStateChangedEvent) -> None:
            asyncio.create_task(
                self._metrics_collector.on_agent_state_changed(
                    old_state=event.old_state,
                    new_state=event.new_state,
                )
            )

        def error_wrapper(event: ErrorEvent) -> None:
            if self._tool_feedback is not None:
                asyncio.create_task(self._tool_feedback.stop_typing_sound(reason="error"))
            source = type(event.source).__name__
            error_type = error_type_name(event.error)
            recoverable = error_recoverable(event.error)
            detail = error_detail(event.error)
            logger.error(
                "Agent session pipeline error: room=%s job_id=%s source=%s error_type=%s recoverable=%s detail=%s",
                self._room_name,
                self._job_id,
                source,
                error_type,
                recoverable,
                detail,
            )

        def close_wrapper(event: CloseEvent) -> None:
            if self._tool_feedback is not None:
                asyncio.create_task(
                    self._tool_feedback.stop_typing_sound(reason=f"close:{event.reason.value}")
                )
            reason = event.reason.value
            if event.error is None:
                logger.info(
                    "Agent session closed: room=%s job_id=%s reason=%s",
                    self._room_name,
                    self._job_id,
                    reason,
                )
                return

            error_type = error_type_name(event.error)
            recoverable = error_recoverable(event.error)
            detail = error_detail(event.error)
            logger.warning(
                "Agent session closed with error: room=%s job_id=%s reason=%s error_type=%s recoverable=%s detail=%s",
                self._room_name,
                self._job_id,
                reason,
                error_type,
                recoverable,
                detail,
            )

        self.session.on("metrics_collected", metrics_wrapper)
        self.session.on("user_input_transcribed", transcript_wrapper)
        self.session.on("conversation_item_added", conversation_item_wrapper)
        self.session.on("speech_created", speech_created_wrapper)
        self.session.on("function_tools_executed", function_tools_executed_wrapper)
        self.session.on("agent_state_changed", agent_state_changed_wrapper)
        self.session.on("error", error_wrapper)
        self.session.on("close", close_wrapper)
