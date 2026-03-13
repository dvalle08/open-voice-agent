import json
from pathlib import Path
from typing import Optional

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from src.core.logger import logger

BASE_DIR = Path(__file__).parent.parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=True)
logger.info(f"Loaded environment from: {ENV_FILE}")

SENSITIVE_KEY_MARKERS = ("key", "token", "secret", "password")
OLLAMA_LOCAL_BASE_URL = "http://localhost:11434/v1"
OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"


def _is_sensitive_key(key: str) -> bool:
    key_lower = key.lower()
    return any(marker in key_lower for marker in SENSITIVE_KEY_MARKERS)


def _redact_sensitive_value(value: object) -> str:
    if value is None:
        return "<not set>"
    if isinstance(value, str) and not value:
        return "<not set>"
    return "<redacted>"


def mask_sensitive_data(data: dict) -> dict:
    masked = {}

    for key, value in data.items():
        if _is_sensitive_key(key):
            masked[key] = _redact_sensitive_value(value)
            continue

        if isinstance(value, dict):
            masked[key] = mask_sensitive_data(value)
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
    TTS_PROVIDER: str = Field(
        default="pocket",
        description="TTS provider: 'pocket', 'deepgram', or 'nvidia'",
    )
    DEEPGRAM_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Shared Deepgram API key for STT/TTS when STT_PROVIDER=deepgram "
            "or TTS_PROVIDER=deepgram"
        ),
    )
    NVIDIA_TTS_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Optional NVIDIA API key override for TTS when TTS_PROVIDER=nvidia; "
            "falls back to NVIDIA_API_KEY when unset"
        ),
    )
    NVIDIA_TTS_VOICE: str = Field(
        default="Magpie-Multilingual.EN-US.Leo",
        description="Default NVIDIA Riva TTS voice name",
    )
    NVIDIA_TTS_LANGUAGE_CODE: str = Field(
        default="en-US",
        description="Language code for NVIDIA TTS",
    )
    NVIDIA_TTS_SERVER: str = Field(
        default="grpc.nvcf.nvidia.com:443",
        description="NVIDIA Riva TTS server address",
    )
    NVIDIA_TTS_FUNCTION_ID: str = Field(
        default="877104f7-e885-42b9-8de8-f6e4c6303969",
        description="NVIDIA Cloud Functions function ID for TTS",
    )
    NVIDIA_TTS_USE_SSL: bool = Field(
        default=True,
        description="Use SSL/TLS for NVIDIA TTS requests",
    )
    POCKET_TTS_VOICE: str = Field(
        default="alba",
        description="Default voice (alba, marius, javert, jean, fantine, cosette, eponine, azelma) or path to audio file",
    )
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
    POCKET_TTS_CONN_TIMEOUT_SEC: float = Field(
        default=45.0,
        gt=0.0,
        le=300.0,
        description="Pocket TTS synthesis timeout in seconds for one request attempt",
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
        default=60,
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
        default=0.18,
        ge=0.1,
        le=1.0,
        description="Minimum speech duration (seconds) before VAD activation",
    )
    VAD_MIN_SILENCE_DURATION: float = Field(
        default=0.55,
        ge=0.1,
        le=2.0,
        description="Minimum silence duration (seconds) before VAD deactivation",
    )
    VAD_THRESHOLD: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="VAD activation threshold (higher = less sensitive, 0.5 is Silero default)",
    )
    MIN_ENDPOINTING_DELAY: float = Field(
        default=0.5,
        ge=0.0,
        le=10.0,
        description=(
            "Minimum endpointing delay (seconds) before committing user turn; "
            "slightly higher values reduce false turn splits"
        ),
    )
    MAX_ENDPOINTING_DELAY: float = Field(
        default=3.0,
        ge=0.1,
        le=10.0,
        description="Maximum endpointing delay (seconds) when turn detector expects continuation",
    )
    PREEMPTIVE_GENERATION: bool = Field(
        default=False,
        description="Enable speculative LLM/TTS generation before final turn commit",
    )

    @model_validator(mode="after")
    def validate_tts_settings(self) -> "VoiceSettings":
        provider = (self.TTS_PROVIDER or "").strip().lower()
        if provider not in {"pocket", "deepgram", "nvidia"}:
            raise ValueError("TTS_PROVIDER must be either 'pocket', 'deepgram', or 'nvidia'")

        self.TTS_PROVIDER = provider

        return self


class STTSettings(CoreSettings):
    # Provider selection
    STT_PROVIDER: str = Field(
        default="deepgram",
        description="STT provider: 'moonshine', 'nvidia', or 'deepgram'"
    )

    # Moonshine STT settings
    MOONSHINE_MODEL_ID: str = Field(
        default="usefulsensors/moonshine-streaming-medium",
        description="Moonshine model size: tiny, base, small, or medium"
    )
    MOONSHINE_LANGUAGE: str = Field(
        default="en",
        description="Language code for Moonshine STT"
    )

    # NVIDIA STT settings
    NVIDIA_STT_API_KEY: Optional[str] = Field(
        default=None,
        description="NVIDIA API key for STT (falls back to NVIDIA_API_KEY if not set)"
    )
    NVIDIA_STT_MODEL: str = Field(
        default="parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
        description="NVIDIA STT model ID"
    )
    NVIDIA_STT_LANGUAGE_CODE: str = Field(
        default="en-US",
        description="Language code for NVIDIA STT"
    )
    DEEPGRAM_STT_MODEL: str = Field(
        default="nova-3",
        description="Deepgram STT model ID",
    )
    DEEPGRAM_STT_LANGUAGE: str = Field(
        default="en-US",
        description="Language code for Deepgram STT",
    )

    @model_validator(mode="after")
    def validate_stt_settings(self) -> "STTSettings":
        provider = (self.STT_PROVIDER or "").strip().lower()
        if provider not in {"moonshine", "nvidia", "deepgram"}:
            raise ValueError("STT_PROVIDER must be either 'moonshine', 'nvidia', or 'deepgram'")

        self.STT_PROVIDER = provider
        return self


class LLMSettings(CoreSettings):
    # Provider selection
    LLM_PROVIDER: str = Field(
        default="ollama",
        description="LLM provider: 'nvidia' or 'ollama'"
    )
    MCP_ENABLED: bool = Field(
        default=True,
        description=(
            "Enable LiveKit MCP runtime. "
            "When enabled, agent sessions expose tools from MCP_SERVER_URL and MCP_EXTRA_SERVER_URLS "
            "for supported providers."
        ),
    )
    MCP_SERVER_URL: str = Field(
        default="https://huggingface.co/mcp",
        description="Primary MCP server URL used by the LiveKit MCP runtime",
    )
    MCP_EXTRA_SERVER_URLS: str = Field(
        default="https://docs.livekit.io/mcp",
        description=(
            "Comma-separated extra MCP server URLs. "
            "Set empty to disable extra MCP servers."
        ),
    )

    # NVIDIA settings
    NVIDIA_API_KEY: Optional[str] = Field(default=None)
    NVIDIA_MODEL: str = Field(default="qwen/qwen3-next-80b-a3b-instruct") #meta/llama-3.1-8b-instruct #"qwen/qwen3-next-80b-a3b-instruct", "qwen/qwen3.5-397b-a17b"

    # Ollama settings
    OLLAMA_CLOUD_MODE: bool = Field(
        default=True,
        description=(
            "Use Ollama Cloud OpenAI-compatible endpoint. "
            "When false, use the local Ollama endpoint."
        ),
    )
    OLLAMA_MODEL: str = Field(
        default= "ministral-3:14b", #"ministral-3:14b-cloud", #"ministral-3:8b-cloud", #"qwen3-coder-next",#minimax-m2.5 #"ministral-3:8b", #"qwen2.5:7b" #"qwen3:8b" #"qwen3.5:4b",
        description="Ollama model tag",
    )
    OLLAMA_API_KEY: Optional[str] = Field(
        default="ollama",
        description="Dummy API key for OpenAI-compatible clients (Ollama ignores auth by default)",
    )

    # Common LLM parameters
    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=256, gt=0)
    LLM_CONN_TIMEOUT_SEC: float = Field(
        default=20.0,
        gt=0.0,
        le=120.0,
        description="LLM API timeout in seconds for one request attempt",
    )
    LLM_CONN_MAX_RETRY: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Maximum LLM retry attempts on transient failures",
    )
    LLM_CONN_RETRY_INTERVAL_SEC: float = Field(
        default=1.0,
        ge=0.0,
        le=30.0,
        description="Delay in seconds between LLM retries",
    )
    TURN_LLM_STALL_TIMEOUT_SEC: float = Field(
        default=12.0,
        gt=0.0,
        le=120.0,
        description="Warn when a finalized user turn does not reach LLM stage within this timeout",
    )
    MCP_STARTUP_GREETING_TIMEOUT_SEC: float = Field(
        default=0.0,
        ge=0.0,
        le=300.0,
        description=(
            "Forced timeout in seconds for the MCP startup greeting. "
            "Set to 0 to disable forced interruption."
        ),
    )

    @property
    def OLLAMA_BASE_URL(self) -> str:
        if self.OLLAMA_CLOUD_MODE:
            return OLLAMA_CLOUD_BASE_URL
        return OLLAMA_LOCAL_BASE_URL

    @model_validator(mode="after")
    def validate_ollama_cloud_settings(self) -> "LLMSettings":
        provider = (self.LLM_PROVIDER or "").strip().lower()
        api_key = (self.OLLAMA_API_KEY or "").strip()

        if provider == "ollama" and self.OLLAMA_CLOUD_MODE and not api_key:
            raise ValueError(
                "OLLAMA_API_KEY is required when LLM_PROVIDER=ollama and OLLAMA_CLOUD_MODE=true"
            )

        return self


class LiveKitSettings(CoreSettings):
    LIVEKIT_URL: Optional[str] = Field(default=None)
    LIVEKIT_API_KEY: Optional[str] = Field(default=None)
    LIVEKIT_API_SECRET: Optional[str] = Field(default=None)
    LIVEKIT_AGENT_NAME: str = Field(default="open-voice-agent")
    LIVEKIT_NUM_IDLE_PROCESSES: int = Field(default=0, ge=0)
    LIVEKIT_INITIALIZE_PROCESS_TIMEOUT_SEC: float = Field(
        default=20.0,
        gt=0.0,
        description="Maximum time to wait for a LiveKit idle worker process to initialize",
    )
    LIVEKIT_JOB_MEMORY_WARN_MB: float = Field(
        default=6144,
        gt=0,
        description="Per-job memory warning threshold in MB",
    )


class LangfuseSettings(CoreSettings):
    LANGFUSE_ENABLED: bool = Field(
        default=False,
        description="Enable Langfuse tracing via OTEL exporter",
    )
    LANGFUSE_PUBLIC_KEY: Optional[str] = Field(default=None)
    LANGFUSE_SECRET_KEY: Optional[str] = Field(default=None)
    LANGFUSE_ENVIRONMENT: str = Field(default="development")
    LANGFUSE_HOST: Optional[str] = Field(
        default=None,
        description="Langfuse host URL, e.g. https://cloud.langfuse.com",
    )
    LANGFUSE_BASE_URL: Optional[str] = Field(
        default=None,
        description="Alternative to LANGFUSE_HOST",
    )
    LANGFUSE_PROJECT_ID: Optional[str] = Field(
        default="cmlrbwznk04ogad07cosnpxoh",
        description="Langfuse project ID used to build UI deep links",
    )
    LANGFUSE_PUBLIC_TRACES: bool = Field(
        default=True,
        description="Mark emitted Langfuse traces as public for shareable URLs",
    )
    LANGFUSE_TRACE_FINALIZE_TIMEOUT_MS: float = Field(
        default=8000.0,
        ge=0.0,
        le=10000.0,
        description="Timeout to wait for assistant text before force-finalizing trace",
    )
    LANGFUSE_POST_TOOL_RESPONSE_TIMEOUT_MS: float = Field(
        default=30000.0,
        ge=0.0,
        le=120000.0,
        description=(
            "Timeout to wait for post-tool assistant response before force-finalizing trace; "
            "telemetry only, does not affect live audio latency"
        ),
    )
    LANGFUSE_MAX_PENDING_TRACE_TASKS: int = Field(
        default=200,
        ge=1,
        le=5000,
        description="Maximum queued background trace emission tasks",
    )
    LANGFUSE_TRACE_FLUSH_TIMEOUT_MS: float = Field(
        default=1000.0,
        ge=0.0,
        le=10000.0,
        description="Best-effort tracer flush timeout in milliseconds",
    )
    LANGFUSE_CONTINUATION_COALESCE_WINDOW_MS: float = Field(
        default=1500.0,
        ge=0.0,
        le=10000.0,
        description=(
            "Window to merge an immediately-following continuation into a prior aborted "
            "turn trace; set to 0 to disable"
        ),
    )


class Settings(CoreSettings):
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    livekit: LiveKitSettings = Field(default_factory=LiveKitSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)

    @model_validator(mode="after")
    def validate_cross_provider_settings(self) -> "Settings":
        if self.voice.TTS_PROVIDER == "nvidia" and self.voice.NVIDIA_TTS_USE_SSL:
            tts_api_key = (self.voice.NVIDIA_TTS_API_KEY or "").strip()
            shared_api_key = (self.llm.NVIDIA_API_KEY or "").strip()
            if not tts_api_key and not shared_api_key:
                raise ValueError(
                    "NVIDIA_TTS_API_KEY or NVIDIA_API_KEY is required when "
                    "TTS_PROVIDER=nvidia and NVIDIA_TTS_USE_SSL=true"
                )

        if (
            self.voice.TTS_PROVIDER == "deepgram"
            or self.stt.STT_PROVIDER == "deepgram"
        ) and not (self.voice.DEEPGRAM_API_KEY or "").strip():
            raise ValueError(
                "DEEPGRAM_API_KEY is required when TTS_PROVIDER=deepgram "
                "or STT_PROVIDER=deepgram"
            )

        return self


try:
    settings = Settings()

    settings_dict = settings.model_dump()
    masked_settings = mask_sensitive_data(settings_dict)
    logger.info(f"Settings loaded: {json.dumps(masked_settings, indent=2)}")

except ValidationError as e:
    safe_errors = e.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )
    logger.exception(
        "Error validating settings: %s",
        json.dumps(safe_errors),
    )
    raise
