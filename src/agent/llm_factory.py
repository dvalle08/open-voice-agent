from huggingface_hub import InferenceClient
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from src.core.logger import logger
from src.core.settings import settings


class LLMFactory:
    @staticmethod
    def create_nvidia_llm(
        model: str = settings.llm.NVIDIA_MODEL,
        temperature: float = settings.llm.LLM_TEMPERATURE,
        max_tokens: int = settings.llm.LLM_MAX_TOKENS,
    ) -> ChatNVIDIA:
        logger.info(f"Initializing NVIDIA LLM: {model}")

        if not settings.llm.NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY must be set to use the NVIDIA LLM provider.")

        return ChatNVIDIA(
            model=model,
            api_key=settings.llm.NVIDIA_API_KEY,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )

    @staticmethod
    def create_huggingface_llm(
        model_id: str,
        provider: str = "auto",
        temperature: float = settings.llm.LLM_TEMPERATURE,
        max_tokens: int = settings.llm.LLM_MAX_TOKENS,
    ) -> ChatHuggingFace:
        token = (settings.llm.HF_TOKEN or "").strip()
        if not token:
            raise ValueError("HF_TOKEN must be set to use the HuggingFace LLM provider.")

        logger.info(f"Initializing HuggingFace LLM: {model_id} via provider={provider}")

        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            provider=provider,
            huggingfacehub_api_token=token,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        return ChatHuggingFace(llm=llm)

    @staticmethod
    def create_huggingface_stt(model_id: str | None = None) -> InferenceClient:
        token = (settings.llm.HF_TOKEN or "").strip()
        if not token:
            raise ValueError("HF_TOKEN must be set to use the HuggingFace STT provider.")

        logger.info(f"Initializing HuggingFace STT: {model_id or 'default'}")

        return InferenceClient(model=model_id, token=token)

    @staticmethod
    def create_huggingface_tts(model_id: str | None = None) -> InferenceClient:
        token = (settings.llm.HF_TOKEN or "").strip()
        if not token:
            raise ValueError("HF_TOKEN must be set to use the HuggingFace TTS provider.")

        logger.info(f"Initializing HuggingFace TTS: {model_id or 'default'}")

        return InferenceClient(model=model_id, token=token)
