from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class ConversationState(TypedDict):
    messages: list[BaseMessage]
    current_transcript: str
    context: dict[str, Any]
    turn_active: bool
