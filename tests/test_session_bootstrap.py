from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.api import session_bootstrap
from src.api.livekit_tokens import LiveKitToken
from src.core.settings import settings


def _make_dispatch(dispatch_id: str, worker_id: str | None) -> Any:
    jobs = []
    if worker_id is not None:
        jobs.append(SimpleNamespace(state=SimpleNamespace(worker_id=worker_id)))
    return SimpleNamespace(id=dispatch_id, state=SimpleNamespace(jobs=jobs))


def test_build_session_bootstrap_payload_returns_fresh_room_and_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_calls: list[dict[str, Any]] = []

    def fake_create_room_token(*, room_name: str) -> LiveKitToken:
        return LiveKitToken(
            token=f"token-{room_name}",
            room_name=room_name,
            identity=f"web-{room_name}",
        )

    def fake_ensure_dispatch(
        *,
        room_name: str,
        agent_name: str,
        metadata: str | None = None,
        reset_existing: bool = False,
    ) -> Any:
        captured_calls.append(
            {
                "room_name": room_name,
                "agent_name": agent_name,
                "metadata": metadata,
                "reset_existing": reset_existing,
            }
        )
        return _make_dispatch(dispatch_id=f"dispatch-{room_name}", worker_id="worker-123")

    monkeypatch.setattr(session_bootstrap, "create_room_token", fake_create_room_token)
    monkeypatch.setattr(session_bootstrap, "ensure_agent_dispatched_sync", fake_ensure_dispatch)

    first = session_bootstrap.build_session_bootstrap_payload()
    second = session_bootstrap.build_session_bootstrap_payload()

    assert first.room_name.startswith("voice-")
    assert second.room_name.startswith("voice-")
    assert first.room_name != second.room_name
    assert first.session_id != second.session_id
    assert first.token == f"token-{first.room_name}"
    assert second.token == f"token-{second.room_name}"
    assert first.dispatch_worker_id == "worker-123"
    assert second.dispatch_worker_id == "worker-123"

    assert len(captured_calls) == 2
    for payload, call in ((first, captured_calls[0]), (second, captured_calls[1])):
        assert call["agent_name"] == settings.livekit.LIVEKIT_AGENT_NAME
        assert call["reset_existing"] is True
        assert isinstance(call["metadata"], str)
        metadata = json.loads(call["metadata"])
        assert metadata["type"] == "session_meta"
        assert metadata["session_id"] == payload.session_id
        assert metadata["participant_id"] == payload.participant_identity


def test_build_session_bootstrap_payload_handles_missing_worker_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_bootstrap,
        "create_room_token",
        lambda *, room_name: LiveKitToken(
            token="token",
            room_name=room_name,
            identity="web-test",
        ),
    )
    monkeypatch.setattr(
        session_bootstrap,
        "ensure_agent_dispatched_sync",
        lambda **_: _make_dispatch(dispatch_id="dispatch-1", worker_id=None),
    )

    payload = session_bootstrap.build_session_bootstrap_payload()

    assert payload.dispatch_id == "dispatch-1"
    assert payload.dispatch_worker_id is None
