import json
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from src.core.logger import logger

BASE_DIR = Path(__file__).parent.parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=True)
logger.info(f"Loaded environment from: {ENV_FILE}")


def mask_sensitive_data(data: dict[str, Any]) -> dict[str, Any]:
    masked = {}
    sensitive_keys = ["key", "token", "secret", "password"]
    
    for key, value in data.items():
        if isinstance(value, dict):
            masked[key] = mask_sensitive_data(value)
        elif isinstance(value, str) and any(s in key.lower() for s in sensitive_keys):
            if not value:
                masked[key] = "<not set>"
            elif len(value) <= 4:
                masked[key] = "***"
            else:
                masked[key] = f"{value[:4]}...{value[-4:]}"
        else:
            masked[key] = value
    
    return masked


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        protected_namespaces=(),
    )


class VoiceSettings(CoreSettings):
    VOICE_PROVIDER: str = Field(default="nvidia")

    NVIDIA_VOICE_LANGUAGE: str = Field(default="en-US")
    NVIDIA_VOICE_NAME: str = Field(default="Magpie-Multilingual.EN-US.Aria")
    NVIDIA_TTS_MODEL: str = Field(default="magpie-tts-multilingual")
    NVIDIA_TTS_ENDPOINT: str = Field(default="")

    SAMPLE_RATE_OUTPUT: int = Field(default=48000, gt=0)
    CHUNK_DURATION_MS: int = Field(default=80, gt=0)

    VAD_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    VAD_HORIZON_INDEX: int = Field(default=2, ge=0)


class LLMSettings(CoreSettings):
    NVIDIA_API_KEY: Optional[str] = Field(default=None)
    NVIDIA_MODEL: str = Field(default="meta/llama-3.1-8b-instruct")
    NVIDIA_BASE_URL: str = Field(default="https://integrate.api.nvidia.com/v1")

    HF_TOKEN: Optional[str] = Field(default=None)

    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=1024, gt=0)


class APISettings(CoreSettings):
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, gt=0, lt=65536)
    API_WORKERS: int = Field(default=1, gt=0)
    API_CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000"]
    )


class Settings(CoreSettings):
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)


try:
    settings = Settings()
    
    settings_dict = settings.model_dump()
    masked_settings = mask_sensitive_data(settings_dict)
    logger.info(f"Settings loaded: {json.dumps(masked_settings, indent=2)}")
    
except ValidationError as e:
    logger.exception(f"Error validating settings: {e.json()}")
    raise
