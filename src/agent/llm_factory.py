from typing import Dict, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from src.core.logger import logger
from src.core.settings import settings


class LLMFactory:
    _instances: Dict[str, BaseLanguageModel] = {}

    @classmethod
    def create_llm(cls, provider: Optional[str] = None) -> BaseLanguageModel:
        provider = (provider or settings.llm.LLM_PROVIDER).lower()

        if provider in cls._instances:
            return cls._instances[provider]

        if provider == "nvidia":
            llm = cls._create_nvidia_llm()
        elif provider == "huggingface":
            llm = cls._create_huggingface_llm()
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

    @staticmethod
    def _create_nvidia_llm() -> BaseLanguageModel:
        logger.info(f"Initializing NVIDIA LLM: {settings.llm.NVIDIA_MODEL}")

        if not settings.llm.NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY must be set to use the NVIDIA LLM provider.")

        return ChatNVIDIA(
            model=settings.llm.NVIDIA_MODEL,
            api_key=settings.llm.NVIDIA_API_KEY,
            temperature=settings.llm.LLM_TEMPERATURE,
            max_completion_tokens=settings.llm.LLM_MAX_TOKENS,
        )

    @staticmethod
    def _create_huggingface_llm() -> BaseLanguageModel:
        model_id = settings.llm.HF_MODEL
        if not model_id:
            raise ValueError("HF_MODEL must be set when using the HuggingFace LLM provider.")

        if settings.llm.HF_USE_INFERENCE_API:
            if not settings.llm.HF_TOKEN or not settings.llm.HF_TOKEN.strip():
                raise ValueError(
                    "HF_TOKEN must be provided when HF_USE_INFERENCE_API is true."
                )

            logger.info(f"Initializing Hugging Face Inference API LLM: {model_id}")
            return HuggingFaceEndpoint(
                repo_id=model_id,
                huggingfacehub_api_token=settings.llm.HF_TOKEN,
                temperature=settings.llm.LLM_TEMPERATURE,
                max_new_tokens=settings.llm.LLM_MAX_TOKENS,
            )

        logger.info(f"Initializing local Hugging Face LLM: {model_id}")
        logger.info("Downloading model if not already cached...")
        return HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            trust_remote_code=settings.llm.HF_TRUST_REMOTE_CODE,
            device_map="auto",
            model_kwargs={
                "temperature": settings.llm.LLM_TEMPERATURE,
                "do_sample": True,
            },
            pipeline_kwargs={
                "max_new_tokens": settings.llm.LLM_MAX_TOKENS,
                "temperature": settings.llm.LLM_TEMPERATURE,
                "do_sample": True,
                "tokenizer_kwargs": {
                    "use_fast": settings.llm.HF_USE_FAST_TOKENIZER,
                },
            },
        )
