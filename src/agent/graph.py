"""LangGraph conversation graph with embedded node functions."""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import ConversationState
from src.core.logger import logger
from src.core.settings import settings


def _initialize_llm():
    """Initialize the LLM based on settings."""
    provider = settings.llm.LLM_PROVIDER.lower()
    
    if provider == "nvidia":
        logger.info(f"Initializing NVIDIA LLM: {settings.llm.NVIDIA_MODEL}")
        return ChatNVIDIA(
            model=settings.llm.NVIDIA_MODEL,
            api_key=settings.llm.NVIDIA_API_KEY,
            temperature=settings.llm.LLM_TEMPERATURE,
            max_tokens=settings.llm.LLM_MAX_TOKENS,
        )
    elif provider == "huggingface":
        logger.info(f"Initializing Hugging Face LLM: {settings.llm.HF_MODEL}")
        return HuggingFaceEndpoint(
            repo_id=settings.llm.HF_MODEL,
            huggingfacehub_api_token=settings.llm.HF_TOKEN,
            temperature=settings.llm.LLM_TEMPERATURE,
            max_new_tokens=settings.llm.LLM_MAX_TOKENS,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# Initialize LLM at module level
llm = _initialize_llm()


# System prompt for the voice agent
SYSTEM_PROMPT = """You are a helpful AI voice assistant. You engage in natural, conversational dialogue with users.

Guidelines:
- Keep responses concise and natural for voice interaction
- Be friendly and engaging
- Ask clarifying questions when needed
- Acknowledge what the user says before responding
- Keep your responses focused and to the point (2-3 sentences typically)
"""


def process_user_input(state: ConversationState) -> ConversationState:
    """Process transcribed user input and add to message history.
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with user message added
    """
    transcript = state.get("current_transcript", "").strip()
    
    if not transcript:
        logger.debug("No transcript to process")
        return state
    
    logger.info(f"Processing user input: {transcript}")
    
    # Add user message to history
    messages = state.get("messages", [])
    
    # Add system message if this is the first message
    if not messages:
        messages.append(SystemMessage(content=SYSTEM_PROMPT))
    
    messages.append(HumanMessage(content=transcript))
    
    return {
        **state,
        "messages": messages,
        "current_transcript": "",  # Clear transcript after processing
    }


def generate_response(state: ConversationState) -> ConversationState:
    """Generate AI response using the LLM.
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with AI response added
    """
    messages = state.get("messages", [])
    
    if not messages:
        logger.warning("No messages to generate response from")
        return state
    
    logger.info("Generating AI response...")
    
    try:
        # Generate response using LLM
        response = llm.invoke(messages)
        
        # Extract content from response
        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)
        
        logger.info(f"Generated response: {content[:100]}...")
        
        # Add AI message to history
        messages.append(AIMessage(content=content))
        
        return {
            **state,
            "messages": messages,
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Return fallback response
        fallback = "I'm sorry, I encountered an error. Could you please repeat that?"
        messages.append(AIMessage(content=fallback))
        return {
            **state,
            "messages": messages,
        }


def should_respond(state: ConversationState) -> Literal["generate", "wait"]:
    """Determine if the agent should generate a response or wait.
    
    Args:
        state: Current conversation state
        
    Returns:
        "generate" to generate response, "wait" to wait for more input
    """
    turn_active = state.get("turn_active", False)
    current_transcript = state.get("current_transcript", "").strip()
    
    # If turn is still active, wait
    if turn_active:
        logger.debug("Turn still active, waiting...")
        return "wait"
    
    # If there's a transcript to process, generate response
    if current_transcript:
        logger.debug("Turn complete with transcript, generating response")
        return "generate"
    
    # Otherwise wait
    logger.debug("No action needed, waiting...")
    return "wait"


def create_conversation_graph() -> StateGraph:
    """Create and compile the conversation graph.
    
    Returns:
        Compiled LangGraph conversation graph
    """
    logger.info("Creating conversation graph...")
    
    # Create graph with state schema
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("process_input", process_user_input)
    workflow.add_node("generate_response", generate_response)
    
    # Set entry point
    workflow.set_entry_point("process_input")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "process_input",
        should_respond,
        {
            "generate": "generate_response",
            "wait": END,
        },
    )
    
    # Add edge from generate to end
    workflow.add_edge("generate_response", END)
    
    # Compile with memory checkpointer for conversation history
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    logger.info("Conversation graph created successfully")
    return graph
