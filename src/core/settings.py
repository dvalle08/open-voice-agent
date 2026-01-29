import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from src.core.logger import logger

BASE_DIR = Path(__file__).parent.parent.parent

load_dotenv(BASE_DIR / ".env", override=False)

ENV_STAGE = os.getenv("OVA_STAGE", "prod")

logger.info(f"Loading environment: {ENV_STAGE}")

ENV_MAP = {
    "dev": BASE_DIR / ".env.dev",
    "prod": BASE_DIR / ".env.prod",
}

env_file_path = ENV_MAP.get(ENV_STAGE, ENV_MAP["prod"])
if env_file_path.exists():
    load_dotenv(env_file_path, override=True)
    logger.debug(f"Loaded environment file: {env_file_path}")


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(env_file_path) if env_file_path.exists() else None,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        protected_namespaces=(),
    )


class VoiceSettings(CoreSettings):
    VOICE_PROVIDER: str = Field(default="gradium")
    
    GRADIUM_API_KEY: str = Field(default="")
    GRADIUM_VOICE_ID: str = Field(default="YTpq7expH9539ERJ")
    GRADIUM_MODEL_NAME: str = Field(default="default")
    GRADIUM_REGION: str = Field(default="eu")
    GRADIUM_OUTPUT_FORMAT: str = Field(default="wav")
    GRADIUM_INPUT_FORMAT: str = Field(default="pcm")
    
    SAMPLE_RATE_OUTPUT: int = Field(default=48000, gt=0)
    SAMPLE_RATE_INPUT: int = Field(default=24000, gt=0)
    CHUNK_DURATION_MS: int = Field(default=80, gt=0)
    
    VAD_THRESHOLD: float = Field(default=0.5, ge=0.0, le=1.0)
    VAD_HORIZON_INDEX: int = Field(default=2, ge=0)
    
    @field_validator("GRADIUM_REGION")
    @classmethod
    def validate_region(cls, v: str) -> str:
        if v.lower() not in ["eu", "us"]:
            raise ValueError("GRADIUM_REGION must be 'eu' or 'us'")
        return v.lower()


class LLMSettings(CoreSettings):
    LLM_PROVIDER: str = Field(default="nvidia")
    
    NVIDIA_API_KEY: Optional[str] = Field(default=None)
    NVIDIA_MODEL: str = Field(default="meta/llama-3.1-8b-instruct")
    NVIDIA_BASE_URL: str = Field(default="https://integrate.api.nvidia.com/v1")
    
    HF_TOKEN: Optional[str] = Field(default=None)
    HF_MODEL: Optional[str] = Field(default=None)
    
    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=1024, gt=0)
    LLM_STREAMING: bool = Field(default=True)
    
    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v.lower() not in ["nvidia", "huggingface"]:
            raise ValueError("LLM_PROVIDER must be 'nvidia' or 'huggingface'")
        return v.lower()


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
    if ENV_STAGE == "dev":
        logger.debug(f"Settings loaded: {settings.model_dump_json(indent=2)}")
except ValidationError as e:
    logger.exception(f"Error validating settings: {e.json()}")
    raise
