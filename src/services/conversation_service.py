from typing import Optional

from langgraph.graph import CompiledStateGraph

from src.agent.state import ConversationState
from src.core.logger import logger
from src.core.exceptions import LLMError


class ConversationService:
    def __init__(self, conversation_graph: CompiledStateGraph):
        self._graph = conversation_graph
    
    async def process_message(
        self,
        transcript: str,
        session_id: str,
        current_state: Optional[ConversationState] = None,
    ) -> ConversationState:
        if current_state is None:
            current_state = {
                "messages": [],
                "current_transcript": "",
                "context": {},
                "turn_active": False,
            }
        
        current_state["current_transcript"] = transcript
        current_state["turn_active"] = False
        
        try:
            result = self._graph.invoke(
                current_state,
                config={"configurable": {"thread_id": session_id}},
            )
            return result
        except Exception as e:
            logger.error(f"Conversation processing error: {e}")
            raise LLMError(f"Failed to process conversation: {str(e)}") from e
    
    def get_last_response(self, state: ConversationState) -> Optional[str]:
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                return last_message.content
        return None
