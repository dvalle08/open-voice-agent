import json
from pathlib import Path
from typing import Optional

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from src.core.logger import logger

BASE_DIR = Path(__file__).parent.parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=True)
logger.info(f"Loaded environment from: {ENV_FILE}")


def mask_sensitive_data(data: dict) -> dict:
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
    MOONSHINE_MODEL_ID: str = Field(
        default="usefulsensors/moonshine-streaming-medium",
        description="Moonshine model size: tiny, base, or small",
    )
    POCKET_TTS_VOICE: str = Field(
        default="alba",
        description="Default voice (alba, marius, javert, jean, fantine, cosette, eponine, azelma) or path to audio file",
    )
    SAMPLE_RATE_OUTPUT: int = Field(default=48000, gt=0)
    POCKET_TTS_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation",
    )
    POCKET_TTS_LSD_DECODE_STEPS: int = Field(
        default=1,
        ge=1,
        description="LSD decoding steps (higher = better quality, slower)",
    )

    # LiveKit Audio Input Settings
    LIVEKIT_SAMPLE_RATE: int = Field(
        default=24000,
        description="Audio input sample rate (Hz)",
    )
    LIVEKIT_NUM_CHANNELS: int = Field(
        default=1,
        description="Number of audio input channels (1=mono)",
    )
    LIVEKIT_FRAME_SIZE_MS: int = Field(
        default=20,
        ge=10,
        le=100,
        description="Audio frame size in milliseconds (smaller = faster VAD response)",
    )
    LIVEKIT_PRE_CONNECT_AUDIO: bool = Field(
        default=True,
        description="Pre-connect audio before room join",
    )
    LIVEKIT_PRE_CONNECT_TIMEOUT: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Timeout for pre-connect audio (seconds)",
    )

    # Voice Activity Detection Settings
    VAD_MIN_SPEECH_DURATION: float = Field(
        default=0.25,
        ge=0.1,
        le=1.0,
        description="Minimum speech duration (seconds) before VAD activation",
    )
    VAD_MIN_SILENCE_DURATION: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Minimum silence duration (seconds) before VAD deactivation",
    )
    VAD_THRESHOLD: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="VAD activation threshold (higher = less sensitive, 0.5 is Silero default)",
    )


class LLMSettings(CoreSettings):
    # Provider selection
    LLM_PROVIDER: str = Field(
        default="huggingface",
        description="LLM provider: 'nvidia' or 'huggingface'"
    )

    # NVIDIA settings (existing)
    NVIDIA_API_KEY: Optional[str] = Field(default=None)
    NVIDIA_MODEL: str = Field(default="qwen/qwen2.5-7b-instruct")

    # HuggingFace settings (new)
    HUGGINGFACE_MODEL_ID: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="HuggingFace model repository ID"
    )
    HUGGINGFACE_DEVICE: Optional[str] = Field(
        default=None,
        description="Device for inference: 'cuda', 'cpu', or None for auto-detect"
    )

    # Common LLM parameters
    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=1024, gt=0)


class LiveKitSettings(CoreSettings):
    LIVEKIT_URL: Optional[str] = Field(default=None)
    LIVEKIT_API_KEY: Optional[str] = Field(default=None)
    LIVEKIT_API_SECRET: Optional[str] = Field(default=None)
    LIVEKIT_AGENT_NAME: str = Field(default="open-voice-agent")
    LIVEKIT_NUM_IDLE_PROCESSES: int = Field(default=1, ge=0)


class Settings(CoreSettings):
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    livekit: LiveKitSettings = Field(default_factory=LiveKitSettings)


try:
    settings = Settings()

    settings_dict = settings.model_dump()
    masked_settings = mask_sensitive_data(settings_dict)
    logger.info(f"Settings loaded: {json.dumps(masked_settings, indent=2)}")

except ValidationError as e:
    logger.exception(f"Error validating settings: {e.json()}")
    raise
