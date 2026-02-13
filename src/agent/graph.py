from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, MessagesState, START, END
from livekit.plugins import nvidia

from src.core.settings import settings
from src.core.logger import logger
from src.plugins.huggingface_llm import HuggingFaceLLM
from src.plugins.moonshine_stt import MoonshineSTT


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


def create_stt():
    """Create an STT instance based on the configured provider."""
    provider = settings.stt.STT_PROVIDER.lower()

    if provider == "nvidia":
        logger.info(
            f"Initializing NVIDIA STT: {settings.stt.NVIDIA_STT_MODEL} "
            f"(language: {settings.stt.NVIDIA_STT_LANGUAGE_CODE})"
        )
        # Use NVIDIA_STT_API_KEY if set, otherwise fall back to NVIDIA_API_KEY
        api_key = settings.stt.NVIDIA_STT_API_KEY or settings.llm.NVIDIA_API_KEY
        return nvidia.STT(
            language_code=settings.stt.NVIDIA_STT_LANGUAGE_CODE,
            model=settings.stt.NVIDIA_STT_MODEL,
            api_key=api_key,
        )

    elif provider == "moonshine":
        logger.info(
            f"Initializing Moonshine STT: {settings.stt.MOONSHINE_MODEL_ID} "
            f"(language: {settings.stt.MOONSHINE_LANGUAGE})"
        )
        return MoonshineSTT(
            model_id=settings.stt.MOONSHINE_MODEL_ID,
            language=settings.stt.MOONSHINE_LANGUAGE,
        )

    else:
        raise ValueError(
            f"Unknown STT provider: {provider}. Must be 'nvidia' or 'moonshine'"
        )


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
