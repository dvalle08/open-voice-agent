from typing import Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_huggingface import HuggingFaceEndpoint
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from src.core.logger import logger
from src.core.settings import settings


class LLMFactory:
    _instance: Optional[BaseLanguageModel] = None
    
    @classmethod
    def create_llm(cls, use_cache: bool = True) -> BaseLanguageModel:
        if use_cache and cls._instance is not None:
            return cls._instance
        
        provider = settings.llm.LLM_PROVIDER.lower()
        
        if provider == "nvidia":
            logger.info(f"Initializing NVIDIA LLM: {settings.llm.NVIDIA_MODEL}")
            llm = ChatNVIDIA(
                model=settings.llm.NVIDIA_MODEL,
                api_key=settings.llm.NVIDIA_API_KEY,
                temperature=settings.llm.LLM_TEMPERATURE,
                max_completion_tokens=settings.llm.LLM_MAX_TOKENS,
            )
        elif provider == "huggingface":
            logger.info(f"Initializing Hugging Face LLM: {settings.llm.HF_MODEL}")
            llm = HuggingFaceEndpoint(
                repo_id=settings.llm.HF_MODEL,
                huggingfacehub_api_token=settings.llm.HF_TOKEN,
                temperature=settings.llm.LLM_TEMPERATURE,
                max_new_tokens=settings.llm.LLM_MAX_TOKENS,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        if use_cache:
            cls._instance = llm
        
        return llm
    
    @classmethod
    def reset_cache(cls) -> None:
        cls._instance = None
