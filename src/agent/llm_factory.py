from typing import Optional, Dict

from langchain_core.language_models import BaseLanguageModel
from langchain_huggingface import HuggingFaceEndpoint
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from src.core.logger import logger
from src.core.settings import settings


class LLMFactory:
    _instances: Dict[str, BaseLanguageModel] = {}

    @classmethod
    def create_llm(
        cls,
        provider: Optional[str] = None,
    ) -> BaseLanguageModel:
        provider = (provider or settings.llm.LLM_PROVIDER).lower()

        if provider in cls._instances:
            return cls._instances[provider]

        if provider == "nvidia":
            logger.info(f"Initializing NVIDIA LLM: {settings.llm.NVIDIA_MODEL}")
            llm = ChatNVIDIA(
                model=settings.llm.NVIDIA_MODEL,
                api_key=settings.llm.NVIDIA_API_KEY,
                temperature=settings.llm.LLM_TEMPERATURE,
                max_completion_tokens=settings.llm.LLM_MAX_TOKENS,
            )

        elif provider == "huggingface":
            if not settings.llm.HF_MODEL:
                raise ValueError(
                    "HF_MODEL must be set in environment variables when using HuggingFace provider. "
                    "Set HF_MODEL to a valid HuggingFace model repository ID."
                )
            logger.info(f"Initializing Hugging Face LLM: {settings.llm.HF_MODEL}")
            llm = HuggingFaceEndpoint(
                repo_id=settings.llm.HF_MODEL,
                huggingfacehub_api_token=settings.llm.HF_TOKEN,
                temperature=settings.llm.LLM_TEMPERATURE,
                max_new_tokens=settings.llm.LLM_MAX_TOKENS,
            )

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        cls._instances[provider] = llm
        return llm

    @classmethod
    def reset_cache(cls, provider: Optional[str] = None) -> None:
        if provider:
            cls._instances.pop(provider.lower(), None)
        else:
            cls._instances.clear()
