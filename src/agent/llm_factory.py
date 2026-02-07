
from typing import Any, Union

from huggingface_hub import InferenceClient
from transformers import pipeline
#from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

#from kokoro import KPipeline
import torch
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

    # @staticmethod
    # def create_huggingface_llm(
    #     model_id: str,
    #     provider: str = "auto",
    #     temperature: float = settings.llm.LLM_TEMPERATURE,
    #     max_tokens: int = settings.llm.LLM_MAX_TOKENS,
    #     run_local: bool = False,
    # ) -> ChatHuggingFace:
    #     if run_local:
    #         logger.info(f"Initializing local HuggingFace LLM: {model_id}")
    #         llm = HuggingFacePipeline.from_model_id(
    #             model_id=model_id,
    #             task="text-generation",
    #             pipeline_kwargs={
    #                 "temperature": temperature,
    #                 "max_new_tokens": max_tokens,
    #             },
    #         )
    #         return ChatHuggingFace(llm=llm)

    #     token = (settings.llm.HF_TOKEN or "").strip()
    #     if not token:
    #         raise ValueError("HF_TOKEN must be set to use the HuggingFace LLM provider.")

    #     logger.info(f"Initializing HuggingFace LLM: {model_id} via provider={provider}")

    #     llm = HuggingFaceEndpoint(
    #         repo_id=model_id,
    #         provider=provider,
    #         huggingfacehub_api_token=token,
    #         temperature=temperature,
    #         max_new_tokens=max_tokens,
    #     )
    #     return ChatHuggingFace(llm=llm)

    @staticmethod
    def create_huggingface_stt(
        model_id: str | None = None, run_local: bool = False
    ) -> Union[InferenceClient, Any]:
        if run_local:
            logger.info(f"Initializing local HuggingFace STT: {model_id or 'default'}")
            return pipeline("automatic-speech-recognition", model=model_id)

        token = (settings.llm.HF_TOKEN or "").strip()
        if not token:
            raise ValueError("HF_TOKEN must be set to use the HuggingFace STT provider.")

        logger.info(f"Initializing HuggingFace STT: {model_id or 'default'}")

        return InferenceClient(model=model_id, token=token)

    @staticmethod
    def create_huggingface_tts(
        model_id: str | None = None, run_local: bool = False
    ) -> Union[InferenceClient, Any]:
        if run_local:
            logger.info(f"Initializing local HuggingFace TTS: {model_id or 'default'}")
            return pipeline("text-to-speech", model=model_id)

        token = (settings.llm.HF_TOKEN or "").strip()
        if not token:
            raise ValueError("HF_TOKEN must be set to use the HuggingFace TTS provider.")

        logger.info(f"Initializing HuggingFace TTS: {model_id or 'default'}")

        return InferenceClient(model=model_id, token=token)

    @staticmethod
    def create_kokoro_tts(lang_code: str = "a") -> Any:
        if KPipeline is None:
            raise ImportError(
                "kokoro library not found. Please install it (pip install kokoro>=0.9.4) to use Kokoro TTS."
            )

        logger.info(f"Initializing Kokoro TTS Pipeline with lang_code: {lang_code}")
        return KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")

    @staticmethod
    def create_moonshine_stt(
        model_size: str = "base",
        language: str = "en",
    ) -> "MoonshineSTT":
        """Initialize Moonshine ONNX STT plugin.

        Args:
            model_size: "tiny" (26MB) or "base" (57MB), or language variants (e.g., "base-es", "tiny-ar")
            language: Currently only "en" supported

        Returns:
            MoonshineSTT plugin instance
        """
        logger.info(f"Initializing Moonshine ONNX STT: {model_size}")
        from src.plugins.moonshine_stt import MoonshineSTT
        return MoonshineSTT(model_size=model_size, language=language)

    @staticmethod
    def create_pocket_tts(
        voice: str | None = None,
        temperature: float | None = None,
        lsd_decode_steps: int | None = None,
    ) -> "PocketTTS":
        """Initialize Pocket TTS plugin.

        Args:
            voice: Voice name (alba, marius, etc.) or path to audio file.
                   If None, uses settings.voice.POCKET_TTS_VOICE
            temperature: Sampling temperature (0.0-2.0).
                        If None, uses settings.voice.POCKET_TTS_TEMPERATURE
            lsd_decode_steps: LSD decoding steps for quality.
                             If None, uses settings.voice.POCKET_TTS_LSD_DECODE_STEPS

        Returns:
            PocketTTS plugin instance
        """
        from src.plugins.pocket_tts import PocketTTS

        if voice is None:
            voice = settings.voice.POCKET_TTS_VOICE
        if temperature is None:
            temperature = settings.voice.POCKET_TTS_TEMPERATURE
        if lsd_decode_steps is None:
            lsd_decode_steps = settings.voice.POCKET_TTS_LSD_DECODE_STEPS

        logger.info(f"Initializing Pocket TTS: voice={voice}, temp={temperature}, lsd_steps={lsd_decode_steps}")
        return PocketTTS(
            voice=voice,
            temperature=temperature,
            lsd_decode_steps=lsd_decode_steps,
        )
