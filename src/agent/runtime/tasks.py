"""Background task utilities used by agent session lifecycle."""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import Any

from livekit.agents import AgentSession, llm
from livekit.agents.types import APIConnectOptions

from src.agent.models.llm_runtime import MCP_STARTUP_GREETING_TIMEOUT_SEC, run_startup_greeting
from src.core.logger import logger


async def monitor_startup_greeting_handle(
    greeting_handle: Any,
    *,
    timeout_sec: float = MCP_STARTUP_GREETING_TIMEOUT_SEC,
) -> None:
    speech_id = getattr(greeting_handle, "id", None)
    wait_for_playout = getattr(greeting_handle, "wait_for_playout", None)
    interrupt = getattr(greeting_handle, "interrupt", None)

    if not callable(wait_for_playout):
        logger.warning(
            "Startup greeting handle missing wait_for_playout; speech_id=%s",
            speech_id,
        )
        return

    try:
        await asyncio.wait_for(wait_for_playout(), timeout=timeout_sec)
    except TimeoutError:
        logger.warning(
            "MCP startup greeting timed out after %.2fs; interrupting speech_id=%s",
            timeout_sec,
            speech_id,
        )
        if callable(interrupt):
            try:
                interrupt(force=True)
            except Exception as exc:
                logger.warning("Failed to interrupt timed out startup greeting: %s", exc)
    except asyncio.CancelledError:
        if callable(interrupt):
            try:
                interrupt(force=True)
            except Exception as exc:
                logger.warning("Failed to interrupt cancelled startup greeting: %s", exc)
        logger.info("MCP startup greeting monitor cancelled: speech_id=%s", speech_id)
    except Exception as exc:
        logger.warning("MCP startup greeting monitor failed: %s", exc)


def schedule_startup_greeting_task(
    session: AgentSession,
    *,
    mcp_runtime_active: bool,
) -> asyncio.Task[Any] | None:
    greeting_handle = run_startup_greeting(
        session,
        mcp_runtime_active=mcp_runtime_active,
    )
    if greeting_handle is None:
        return None

    speech_id = getattr(greeting_handle, "id", None)
    logger.info(
        "Scheduling startup greeting monitor task: mcp_runtime_active=%s speech_id=%s",
        mcp_runtime_active,
        speech_id,
    )
    task = asyncio.create_task(
        monitor_startup_greeting_handle(greeting_handle),
        name="startup-greeting-monitor",
    )
    setattr(task, "_open_voice_startup_greeting_handle", greeting_handle)

    def _on_done(completed_task: asyncio.Task[Any]) -> None:
        if completed_task.cancelled():
            return
        try:
            exc = completed_task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.warning(f"Startup greeting monitor task failed: {exc}")

    task.add_done_callback(_on_done)
    return task


async def run_llm_warmup(
    *,
    llm_client: Any,
    conn_options: APIConnectOptions,
    provider: str,
    model: str,
) -> None:
    started = monotonic()
    stream: Any | None = None
    got_first_chunk = False

    try:
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="Reply with OK.")
        stream = llm_client.chat(
            chat_ctx=chat_ctx,
            tools=None,
            conn_options=conn_options,
        )
        async for _ in stream:
            got_first_chunk = True
            break
    except asyncio.CancelledError:
        logger.info("LLM warm-up cancelled: provider=%s model=%s", provider, model)
    except Exception as exc:
        logger.warning("LLM warm-up failed: provider=%s model=%s detail=%s", provider, model, exc)
    finally:
        if stream is not None:
            aclose = getattr(stream, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:
                    pass

        elapsed_ms = max((monotonic() - started) * 1000.0, 0.0)
        logger.info(
            "LLM warm-up completed: provider=%s model=%s first_chunk=%s elapsed_ms=%.1f",
            provider,
            model,
            got_first_chunk,
            elapsed_ms,
        )


def schedule_llm_warmup_task(
    *,
    llm_client: Any,
    conn_options: APIConnectOptions,
    provider: str,
    model: str,
) -> asyncio.Task[Any]:
    logger.info("Scheduling LLM warm-up task: provider=%s model=%s", provider, model)
    task = asyncio.create_task(
        run_llm_warmup(
            llm_client=llm_client,
            conn_options=conn_options,
            provider=provider,
            model=model,
        ),
        name="llm-warmup",
    )

    def _on_done(completed_task: asyncio.Task[Any]) -> None:
        if completed_task.cancelled():
            return
        try:
            exc = completed_task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.warning(f"LLM warm-up task failed: {exc}")

    task.add_done_callback(_on_done)
    return task


async def cancel_task_for_shutdown(
    task: asyncio.Task[Any] | None,
    *,
    task_name: str,
    timeout_sec: float = 0.5,
) -> None:
    if task is None or task.done():
        return

    greeting_handle = getattr(task, "_open_voice_startup_greeting_handle", None)
    if greeting_handle is not None:
        interrupt = getattr(greeting_handle, "interrupt", None)
        if callable(interrupt):
            try:
                interrupt(force=True)
            except Exception as exc:
                logger.warning("%s handle interrupt failed during shutdown: %s", task_name, exc)

    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=timeout_sec)
    except asyncio.CancelledError:
        logger.info("%s task cancelled during shutdown", task_name)
    except TimeoutError:
        logger.warning("%s task did not cancel within %.2fs", task_name, timeout_sec)
    except Exception as exc:
        logger.warning("%s task raised during shutdown cancellation: %s", task_name, exc)
