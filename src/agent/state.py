"""Conversation state schema for LangGraph."""

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class ConversationState(TypedDict):
    """State schema for conversation graph.
    
    Attributes:
        messages: LangChain message history (user and assistant messages)
        current_transcript: Current user speech being transcribed
        context: Additional context for the conversation
        turn_active: Whether the user is currently speaking
    """

    messages: list[BaseMessage]
    current_transcript: str
    context: dict[str, Any]
    turn_active: bool
