"""Runtime user-facing utterances and fallback text."""

MCP_STARTUP_GREETING = "Hi, I am Open Voice Agent. How can I help you today?"
TOOL_PRE_SPEECH_FALLBACK = "Let me check that."
DEFAULT_TOOL_FALLBACK_PHRASES: tuple[str, ...] = (
    TOOL_PRE_SPEECH_FALLBACK,
    "One sec, checking now.",
    "I'll look that up.",
    "Checking that for you.",
)
