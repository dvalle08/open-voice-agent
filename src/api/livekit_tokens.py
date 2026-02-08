from __future__ import annotations

import uuid
import asyncio
from dataclasses import dataclass

from livekit import api

from src.core.settings import settings


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
