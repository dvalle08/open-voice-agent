from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, MessagesState, START, END

from src.core.settings import settings


def create_graph():
    """Create a single-node LangGraph workflow using NVIDIA ChatNVIDIA."""
    llm = ChatNVIDIA(
        model=settings.llm.NVIDIA_MODEL,
        api_key=settings.llm.NVIDIA_API_KEY,
        temperature=settings.llm.LLM_TEMPERATURE,
        max_tokens=settings.llm.LLM_MAX_TOKENS,
    )

    def call_model(state: MessagesState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow.compile()
