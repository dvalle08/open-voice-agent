from __future__ import annotations

import uuid
import asyncio
from dataclasses import dataclass

from livekit import api

from src.core.settings import settings
from src.core.logger import logger


@dataclass(frozen=True)
class LiveKitToken:
    token: str
    room_name: str
    identity: str


def _normalize_livekit_url(url: str | None) -> str:
    if not url:
        raise ValueError("LIVEKIT_URL must be set")
    if url.startswith("wss://"):
        return f"https://{url[6:]}"
    if url.startswith("ws://"):
        return f"http://{url[5:]}"
    return url


async def dispatch_agent(
    *,
    room_name: str,
    agent_name: str,
    metadata: str | None = None,
) -> api.AgentDispatch:
    if not room_name:
        raise ValueError("room_name must not be empty")
    if not agent_name:
        raise ValueError("agent_name must not be empty")

    request = api.CreateAgentDispatchRequest(
        agent_name=agent_name,
        room=room_name,
        metadata=metadata or "",
    )
    lkapi = api.LiveKitAPI(
        url=_normalize_livekit_url(settings.livekit.LIVEKIT_URL),
        api_key=settings.livekit.LIVEKIT_API_KEY,
        api_secret=settings.livekit.LIVEKIT_API_SECRET,
    )
    try:
        return await lkapi.agent_dispatch.create_dispatch(request)
    finally:
        await lkapi.aclose()


async def ensure_agent_dispatched(
    *,
    room_name: str,
    agent_name: str,
    metadata: str | None = None,
    reset_existing: bool = False,
) -> api.AgentDispatch:
    """Ensure a dispatch exists for this room+agent.

    Returns an existing dispatch when present; otherwise creates one.
    """
    if not room_name:
        raise ValueError("room_name must not be empty")
    if not agent_name:
        raise ValueError("agent_name must not be empty")

    lkapi = api.LiveKitAPI(
        url=_normalize_livekit_url(settings.livekit.LIVEKIT_URL),
        api_key=settings.livekit.LIVEKIT_API_KEY,
        api_secret=settings.livekit.LIVEKIT_API_SECRET,
    )
    try:
        try:
            dispatches = await lkapi.agent_dispatch.list_dispatch(room_name=room_name)
        except api.TwirpError as exc:
            # Expected when room does not exist yet; create dispatch below.
            if exc.code != api.TwirpErrorCode.NOT_FOUND:
                raise
            dispatches = []

        if not reset_existing:
            for dispatch in dispatches:
                if dispatch.agent_name == agent_name:
                    return dispatch
        else:
            # If a previous agent worker got stuck on this room, stale dispatch/jobs can
            # block the current worker from receiving job requests. Clean up and recreate.
            running_agent_identities: set[str] = set()
            for dispatch in dispatches:
                if dispatch.agent_name and dispatch.agent_name != agent_name:
                    continue
                logger.info(
                    "Resetting existing dispatch id=%s room=%s agent_name=%s",
                    dispatch.id,
                    dispatch.room,
                    dispatch.agent_name,
                )
                for job in getattr(dispatch.state, "jobs", []):
                    state = getattr(job, "state", None)
                    if not state:
                        continue
                    identity = getattr(state, "participant_identity", "")
                    status = getattr(state, "status", None)
                    if identity and status == 1:  # JS_RUNNING
                        running_agent_identities.add(identity)

            if running_agent_identities:
                try:
                    participants = await lkapi.room.list_participants(
                        api.ListParticipantsRequest(room=room_name)
                    )
                    active_identities = {p.identity for p in participants.participants}
                except api.TwirpError as exc:
                    if exc.code != api.TwirpErrorCode.NOT_FOUND:
                        raise
                    active_identities = set()

                for identity in running_agent_identities:
                    if identity not in active_identities:
                        continue
                    try:
                        await lkapi.room.remove_participant(
                            api.RoomParticipantIdentity(room=room_name, identity=identity)
                        )
                        logger.info(
                            "Removed stale agent participant identity=%s room=%s",
                            identity,
                            room_name,
                        )
                    except api.TwirpError as exc:
                        if exc.code != api.TwirpErrorCode.NOT_FOUND:
                            raise

            for dispatch in dispatches:
                if dispatch.agent_name and dispatch.agent_name != agent_name:
                    continue
                try:
                    await lkapi.agent_dispatch.delete_dispatch(
                        dispatch_id=dispatch.id,
                        room_name=room_name,
                    )
                    logger.info(
                        "Deleted stale dispatch id=%s room=%s",
                        dispatch.id,
                        room_name,
                    )
                except api.TwirpError as exc:
                    if exc.code != api.TwirpErrorCode.NOT_FOUND:
                        raise

        created = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=agent_name,
                room=room_name,
                metadata=metadata or "",
            )
        )
        assigned_worker_id = None
        for job in getattr(created.state, "jobs", []):
            state = getattr(job, "state", None)
            if state and getattr(state, "worker_id", None):
                assigned_worker_id = state.worker_id
                break
        logger.info(
            "Ensured dispatch id=%s room=%s agent_name=%s worker_id=%s",
            created.id,
            room_name,
            agent_name,
            assigned_worker_id or "unassigned",
        )
        return created
    finally:
        await lkapi.aclose()


def dispatch_agent_sync(
    *,
    room_name: str,
    agent_name: str,
    metadata: str | None = None,
) -> api.AgentDispatch:
    return asyncio.run(
        dispatch_agent(
            room_name=room_name,
            agent_name=agent_name,
            metadata=metadata,
        )
    )


def ensure_agent_dispatched_sync(
    *,
    room_name: str,
    agent_name: str,
    metadata: str | None = None,
    reset_existing: bool = False,
) -> api.AgentDispatch:
    return asyncio.run(
        ensure_agent_dispatched(
            room_name=room_name,
            agent_name=agent_name,
            metadata=metadata,
            reset_existing=reset_existing,
        )
    )


def create_room_token(room_name: str, identity: str | None = None) -> LiveKitToken:
    if not room_name:
        raise ValueError("room_name must not be empty")

    participant_identity = identity or f"web-{uuid.uuid4().hex}"

    token = (
        api.AccessToken(
            settings.livekit.LIVEKIT_API_KEY,
            settings.livekit.LIVEKIT_API_SECRET,
        )
        .with_identity(participant_identity)
        .with_grants(api.VideoGrants(room=room_name, room_join=True))
        .to_jwt()
    )

    return LiveKitToken(token=token, room_name=room_name, identity=participant_identity)
