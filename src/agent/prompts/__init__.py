"""Prompt package for agent runtime."""

from src.agent.prompts.assistant import ASSISTANT_INSTRUCTIONS
from src.agent.prompts.runtime import (
    DEFAULT_TOOL_FALLBACK_PHRASES,
    MCP_STARTUP_GREETING,
    TOOL_PRE_SPEECH_FALLBACK,
)

__all__ = [
    "ASSISTANT_INSTRUCTIONS",
    "DEFAULT_TOOL_FALLBACK_PHRASES",
    "MCP_STARTUP_GREETING",
    "TOOL_PRE_SPEECH_FALLBACK",
]
