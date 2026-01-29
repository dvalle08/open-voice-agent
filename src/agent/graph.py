from typing import Literal, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.llm_factory import LLMFactory
from src.agent.prompts import get_system_prompt
from src.agent.state import ConversationState
from src.core.logger import logger


def process_user_input(state: ConversationState) -> ConversationState:
    transcript = state.get("current_transcript", "").strip()
    
    if not transcript:
        logger.debug("No transcript to process")
        return state
    
    logger.info(f"Processing user input: {transcript}")
    
    messages = state.get("messages", [])
    
    if not messages:
        messages.append(SystemMessage(content=get_system_prompt()))
    
    messages.append(HumanMessage(content=transcript))
    
    return {
        **state,
        "messages": messages,
        "current_transcript": "",
    }


def generate_response(state: ConversationState, llm: Optional[BaseLanguageModel] = None) -> ConversationState:
    if llm is None:
        llm = LLMFactory.create_llm()
    
    messages = state.get("messages", [])
    
    if not messages:
        logger.warning("No messages to generate response from")
        return state
    
    logger.info("Generating AI response...")
    
    try:
        response = llm.invoke(messages)
        
        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)
        
        logger.info(f"Generated response: {content[:100]}...")
        
        messages.append(AIMessage(content=content))
        
        return {
            **state,
            "messages": messages,
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        fallback = "I'm sorry, I encountered an error. Could you please repeat that?"
        messages.append(AIMessage(content=fallback))
        return {
            **state,
            "messages": messages,
        }


def should_respond(state: ConversationState) -> Literal["generate", "wait"]:
    turn_active = state.get("turn_active", False)
    current_transcript = state.get("current_transcript", "").strip()
    
    if turn_active:
        logger.debug("Turn still active, waiting...")
        return "wait"
    
    if current_transcript:
        logger.debug("Turn complete with transcript, generating response")
        return "generate"
    
    logger.debug("No action needed, waiting...")
    return "wait"


def create_conversation_graph() -> StateGraph:
    logger.info("Creating conversation graph...")
    
    workflow = StateGraph(ConversationState)
    
    workflow.add_node("process_input", process_user_input)
    workflow.add_node("generate_response", generate_response)
    
    workflow.set_entry_point("process_input")
    
    workflow.add_conditional_edges(
        "process_input",
        should_respond,
        {
            "generate": "generate_response",
            "wait": END,
        },
    )
    
    workflow.add_edge("generate_response", END)
    
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    logger.info("Conversation graph created successfully")
    return graph
