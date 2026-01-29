from typing import Dict, Type, Callable

from src.core.logger import logger
from src.core.settings import settings
from src.models.voice.base import BaseVoiceProvider, VoiceProviderConfig
from src.models.voice.gradium import GradiumProvider, GradiumConfig


class VoiceProviderFactory:
    _registry: Dict[str, Callable[[VoiceProviderConfig], BaseVoiceProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseVoiceProvider]) -> None:
        cls._registry[name.lower()] = provider_class
        logger.debug(f"Registered voice provider: {name}")
    
    @classmethod
    def create_provider(cls, provider_name: str = None) -> BaseVoiceProvider:
        provider_name = provider_name or settings.voice.VOICE_PROVIDER
        provider_name = provider_name.lower()
        
        if provider_name not in cls._registry:
            raise ValueError(f"Unknown voice provider: {provider_name}. Available: {list(cls._registry.keys())}")
        
        logger.info(f"Creating voice provider: {provider_name}")
        
        if provider_name == "gradium":
            config = GradiumConfig(
                api_key=settings.voice.GRADIUM_API_KEY,
                voice_id=settings.voice.GRADIUM_VOICE_ID,
                model_name=settings.voice.GRADIUM_MODEL_NAME,
                region=settings.voice.GRADIUM_REGION,
                sample_rate_input=settings.voice.SAMPLE_RATE_INPUT,
                sample_rate_output=settings.voice.SAMPLE_RATE_OUTPUT,
                vad_threshold=settings.voice.VAD_THRESHOLD,
            )
            return cls._registry[provider_name](config)
        
        raise NotImplementedError(f"Configuration for {provider_name} not yet implemented")


VoiceProviderFactory.register("gradium", GradiumProvider)
