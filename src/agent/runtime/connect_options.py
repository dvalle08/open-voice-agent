"""Helpers for LiveKit API connection option wiring."""

from __future__ import annotations

from livekit.agents.types import APIConnectOptions


def build_api_connect_options(
    *,
    timeout_sec: float,
    max_retry: int,
    retry_interval_sec: float,
) -> APIConnectOptions:
    """Build API connect options from runtime tuning values."""
    return APIConnectOptions(
        max_retry=max_retry,
        retry_interval=retry_interval_sec,
        timeout=timeout_sec,
    )
