from __future__ import annotations

import uuid
from dataclasses import dataclass

from livekit import api

from src.core.settings import settings


@dataclass(frozen=True)
class LiveKitToken:
    token: str
    room_name: str
    identity: str


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
    