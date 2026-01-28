"""Environment-based configuration using Pydantic Settings.

Supports multiple environments (dev, prod) via OVA_STAGE environment variable.
Configuration is loaded from .env files with environment-specific overrides.
"""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from src.core.logger import logger

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent.parent

# Load base .env file first
load_dotenv(BASE_DIR / ".env", override=False)

# Determine environment stage
ENV_STAGE = os.getenv("OVA_STAGE", "prod")  # OVA = Open Voice Agent

logger.info(f"Loading environment: {ENV_STAGE}")

ENV_MAP = {
    "dev": BASE_DIR / ".env.dev",
    "prod": BASE_DIR / ".env.prod",
}

# Load environment-specific .env file
env_file_path = ENV_MAP.get(ENV_STAGE, ENV_MAP["prod"])
if env_file_path.exists():
    load_dotenv(env_file_path, override=True)
    logger.debug(f"Loaded environment file: {env_file_path}")


class CoreSettings(BaseSettings):
    """Base settings class with common configuration."""

    model_config = SettingsConfigDict(
        env_file=str(env_file_path) if env_file_path.exists() else None,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        protected_namespaces=(),
    )


class VoiceSettings(CoreSettings):
    """Voice provider configuration."""

    # Provider selection
    VOICE_PROVIDER: str = Field(default="gradium", description="Voice provider to use")

    # Gradium settings
    GRADIUM_API_KEY: str = Field(default="", description="Gradium API key")
    GRADIUM_VOICE_ID: str = Field(
        default="YTpq7expH9539ERJ", description="Gradium voice ID (Emma)"
    )
    GRADIUM_MODEL_NAME: str = Field(default="default", description="Gradium model name")
    GRADIUM_REGION: str = Field(default="eu", description="Gradium region (eu or us)")
    GRADIUM_OUTPUT_FORMAT: str = Field(default="wav", description="Audio output format")
    GRADIUM_INPUT_FORMAT: str = Field(default="pcm", description="Audio input format")

    # Audio settings
    SAMPLE_RATE_OUTPUT: int = Field(default=48000, description="Output sample rate (Hz)")
    SAMPLE_RATE_INPUT: int = Field(default=24000, description="Input sample rate (Hz)")
    CHUNK_DURATION_MS: int = Field(default=80, description="Audio chunk duration (ms)")

    # VAD settings
    VAD_THRESHOLD: float = Field(
        default=0.5, description="Voice activity detection threshold"
    )
    VAD_HORIZON_INDEX: int = Field(
        default=2, description="VAD horizon index for turn detection"
    )


class LLMSettings(CoreSettings):
    """LLM provider configuration."""

    # Provider selection
    LLM_PROVIDER: str = Field(
        default="nvidia", description="LLM provider (nvidia or huggingface)"
    )

    # NVIDIA settings
    NVIDIA_API_KEY: Optional[str] = Field(default=None, description="NVIDIA API key")
    NVIDIA_MODEL: str = Field(
        default="meta/llama-3.1-8b-instruct", description="NVIDIA model name"
    )
    NVIDIA_BASE_URL: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA API base URL",
    )

    # Hugging Face settings
    HF_TOKEN: Optional[str] = Field(default=None, description="Hugging Face API token")
    HF_MODEL: Optional[str] = Field(
        default=None, description="Hugging Face model name"
    )

    # Generation settings
    LLM_TEMPERATURE: float = Field(default=0.7, description="LLM temperature")
    LLM_MAX_TOKENS: int = Field(default=1024, description="Maximum tokens to generate")
    LLM_STREAMING: bool = Field(
        default=True, description="Enable streaming responses"
    )


class APISettings(CoreSettings):
    """API server configuration."""

    API_HOST: str = Field(default="0.0.0.0", description="API server host")
    API_PORT: int = Field(default=8000, description="API server port")
    API_WORKERS: int = Field(default=1, description="Number of API workers")
    API_CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000"],
        description="CORS allowed origins",
    )


class Settings(CoreSettings):
    """Main settings aggregating all sub-settings."""

    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)


# Initialize settings singleton
try:
    settings = Settings()
    if ENV_STAGE == "dev":
        logger.debug(f"Settings loaded: {settings.model_dump_json(indent=2)}")
except ValidationError as e:
    logger.exception(f"Error validating settings: {e.json()}")
    raise
