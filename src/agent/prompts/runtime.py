"""Runtime user-facing utterances and fallback text."""

MCP_STARTUP_GREETING = (
    "Hi, I'm Open Voice Agent. I can chat naturally, explain my current STT, LLM, TTS, and LiveKit setup, or use MCP tools like Hugging Face and LiveKit Docs when needed. "
    "How can I help today?"
)
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
