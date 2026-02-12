from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, MessagesState, START, END

from src.core.settings import settings
from src.core.logger import logger
from src.plugins.huggingface_llm import HuggingFaceLLM


def create_llm():
    """Create an LLM instance based on the configured provider."""
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
        logger.info(f"Initializing HuggingFace LLM: {settings.llm.HUGGINGFACE_MODEL_ID}")
        return HuggingFaceLLM(
            model_id=settings.llm.HUGGINGFACE_MODEL_ID,
            device=settings.llm.HUGGINGFACE_DEVICE,
            temperature=settings.llm.LLM_TEMPERATURE,
            max_tokens=settings.llm.LLM_MAX_TOKENS,
            top_p=0.95,
            repetition_penalty=1.0,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Must be 'nvidia' or 'huggingface'")

def create_graph():
    """Create a single-node LangGraph workflow with the configured LLM provider."""
    llm = create_llm()

    def call_model(state: MessagesState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow.compile()

graph = create_graph()

for msg in graph.stream({"messages": [{"role": "user", "content": "Hello, how are you?"}]}, stream_mode="messages"):
    print(msg)