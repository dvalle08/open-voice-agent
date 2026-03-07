"""Runtime user-facing utterances and fallback text."""

MCP_STARTUP_GREETING = "Hi, I'm Open Voice Agent. How can I help you today?"
TOOL_PRE_SPEECH_FALLBACK = "Let me check that."
DEFAULT_TOOL_FALLBACK_PHRASES: tuple[str, ...] = (
    TOOL_PRE_SPEECH_FALLBACK,
    "One sec, checking now.",
    "I'll look that up.",
    "Checking that for you.",
    "Give me a moment, I'll check that.",
    "Sure, let me pull that up.",
    "Okay, I'll verify that now.",
)
