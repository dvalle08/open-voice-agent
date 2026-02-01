from typing import Dict, Type, Callable

from src.core.logger import logger
from src.core.settings import settings
from src.models.voice.base import BaseVoiceProvider, VoiceProviderConfig
from src.models.voice.gradium import GradiumProvider, GradiumConfig
from src.models.voice.nvidia import NvidiaVoiceProvider, NvidiaConfig


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
        
        if provider_name == "nvidia":
            config = NvidiaConfig(
                api_key=settings.llm.NVIDIA_API_KEY,
                language=settings.voice.NVIDIA_VOICE_LANGUAGE,
                voice_name=settings.voice.NVIDIA_VOICE_NAME,
                asr_model=settings.voice.NVIDIA_ASR_MODEL,
                tts_model=settings.voice.NVIDIA_TTS_MODEL,
                grpc_server=settings.voice.NVIDIA_GRPC_SERVER,
                asr_function_id=settings.voice.NVIDIA_ASR_FUNCTION_ID,
                tts_endpoint=settings.voice.NVIDIA_TTS_ENDPOINT,
                tts_api_type=settings.voice.NVIDIA_TTS_API_TYPE,
                sample_rate_input=settings.voice.SAMPLE_RATE_INPUT,
                sample_rate_output=settings.voice.SAMPLE_RATE_OUTPUT,
            )
            return cls._registry[provider_name](config)
        
        raise NotImplementedError(f"Configuration for {provider_name} not yet implemented")


VoiceProviderFactory.register("gradium", GradiumProvider)
VoiceProviderFactory.register("nvidia", NvidiaVoiceProvider)
