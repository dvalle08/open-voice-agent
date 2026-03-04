from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.agent.traces.channel_metrics import ChannelPublisher


class _FailingLocalParticipant:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def publish_data(
        self,
        *,
        payload: bytes,
        topic: str,
        reliable: bool,
    ) -> None:
        _ = (payload, topic, reliable)
        raise self._exc


class _FakeRoom:
    def __init__(self, exc: Exception) -> None:
        self.local_participant = _FailingLocalParticipant(exc)


def test_publish_live_update_downgrades_preconnect_publish_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.channel_metrics as channel_metrics_module

    debug_logs: list[tuple[Any, ...]] = []
    error_logs: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        channel_metrics_module.logger,
        "debug",
        lambda *args, **kwargs: debug_logs.append((args, kwargs)),
    )
    monkeypatch.setattr(
        channel_metrics_module.logger,
        "error",
        lambda *args, **kwargs: error_logs.append((args, kwargs)),
    )

    room = _FakeRoom(
        RuntimeError("cannot access local participant before connecting")
    )
    publisher = ChannelPublisher(room)  # type: ignore[arg-type]

    asyncio.run(
        publisher.publish_live_update(
            speech_id=None,
            stage="llm",
            role=None,
            turn_metrics=None,
        )
    )

    assert debug_logs
    assert not error_logs


def test_publish_live_update_keeps_error_logging_for_other_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.agent.traces.channel_metrics as channel_metrics_module

    debug_logs: list[tuple[Any, ...]] = []
    error_logs: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        channel_metrics_module.logger,
        "debug",
        lambda *args, **kwargs: debug_logs.append((args, kwargs)),
    )
    monkeypatch.setattr(
        channel_metrics_module.logger,
        "error",
        lambda *args, **kwargs: error_logs.append((args, kwargs)),
    )

    room = _FakeRoom(RuntimeError("unexpected publish failure"))
    publisher = ChannelPublisher(room)  # type: ignore[arg-type]

    asyncio.run(
        publisher.publish_live_update(
            speech_id=None,
            stage="llm",
            role=None,
            turn_metrics=None,
        )
    )

    assert error_logs
    assert not debug_logs
