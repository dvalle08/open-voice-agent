"""Prompt package for agent runtime."""

from src.agent.prompts.assistant import (
    ASSISTANT_INSTRUCTIONS,
    build_assistant_instructions,
)
from src.agent.prompts.runtime import (
    DEFAULT_TOOL_FALLBACK_PHRASES,
    MCP_STARTUP_GREETING,
    TOOL_PRE_SPEECH_FALLBACK,
)

__all__ = [
    "ASSISTANT_INSTRUCTIONS",
    "build_assistant_instructions",
    "DEFAULT_TOOL_FALLBACK_PHRASES",
    "MCP_STARTUP_GREETING",
    "TOOL_PRE_SPEECH_FALLBACK",
]
