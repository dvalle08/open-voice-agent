from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.settings import (
    LLMSettings,
    LangfuseSettings,
    LiveKitSettings,
    STTSettings,
    Settings,
    VoiceSettings,
)


def test_llm_runtime_tuning_defaults_are_declared() -> None:
    fields = LLMSettings.model_fields
    settings = LLMSettings(OLLAMA_API_KEY="test-key")

    assert fields["LLM_PROVIDER"].default == "ollama"
    assert fields["MCP_ENABLED"].default is True
    assert fields["MCP_SERVER_URL"].default == "https://huggingface.co/mcp"
    assert fields["MCP_EXTRA_SERVER_URLS"].default == "https://docs.livekit.io/mcp"
    assert fields["OLLAMA_CLOUD_MODE"].default is True
    assert fields["OLLAMA_MODEL"].default == "ministral-3:14b"
    assert fields["OLLAMA_API_KEY"].default == "ollama"
    assert fields["LLM_CONN_TIMEOUT_SEC"].default == 20.0
    assert fields["LLM_CONN_MAX_RETRY"].default == 1
    assert fields["LLM_CONN_RETRY_INTERVAL_SEC"].default == 1.0
    assert fields["TURN_LLM_STALL_TIMEOUT_SEC"].default == 12.0
    assert fields["MCP_STARTUP_GREETING_TIMEOUT_SEC"].default == 0.0
    assert settings.OLLAMA_BASE_URL == "https://ollama.com/v1"


def test_livekit_runtime_tuning_defaults_are_declared() -> None:
    fields = LiveKitSettings.model_fields

    assert fields["LIVEKIT_NUM_IDLE_PROCESSES"].default == 0
    assert fields["LIVEKIT_INITIALIZE_PROCESS_TIMEOUT_SEC"].default == 20.0
    assert fields["LIVEKIT_JOB_MEMORY_WARN_MB"].default == 6144


def test_voice_runtime_tuning_defaults_are_declared() -> None:
    fields = VoiceSettings.model_fields

    assert fields["TTS_PROVIDER"].default == "pocket"
    assert fields["NVIDIA_TTS_VOICE"].default == "Magpie-Multilingual.EN-US.Leo"
    assert fields["NVIDIA_TTS_USE_SSL"].default is True
    assert fields["POCKET_TTS_CONN_TIMEOUT_SEC"].default == 45.0
    assert fields["MIN_ENDPOINTING_DELAY"].default == 0.8


def test_langfuse_runtime_tuning_defaults_are_declared() -> None:
    fields = LangfuseSettings.model_fields

    assert fields["LANGFUSE_TRACE_FINALIZE_TIMEOUT_MS"].default == 8000.0
    assert fields["LANGFUSE_POST_TOOL_RESPONSE_TIMEOUT_MS"].default == 30000.0
    assert fields["LANGFUSE_CONTINUATION_COALESCE_WINDOW_MS"].default == 1500.0


def test_stt_runtime_tuning_defaults_are_declared() -> None:
    fields = STTSettings.model_fields

    assert fields["STT_PROVIDER"].default == "deepgram"
    assert fields["DEEPGRAM_STT_MODEL"].default == "nova-3"
    assert fields["DEEPGRAM_STT_LANGUAGE"].default == "en-US"


def test_llm_runtime_tuning_switches_to_local_ollama_base_url_when_cloud_mode_disabled() -> None:
    settings = LLMSettings(OLLAMA_CLOUD_MODE=False, OLLAMA_API_KEY=None)

    assert settings.OLLAMA_BASE_URL == "http://localhost:11434/v1"


def test_llm_runtime_tuning_requires_api_key_for_ollama_cloud_mode() -> None:
    with pytest.raises(
        ValidationError,
        match="OLLAMA_API_KEY is required when LLM_PROVIDER=ollama and OLLAMA_CLOUD_MODE=true",
    ):
        LLMSettings(OLLAMA_API_KEY=" ")


def test_llm_runtime_tuning_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        LLMSettings(LLM_CONN_TIMEOUT_SEC=0.0)

    with pytest.raises(ValidationError):
        LLMSettings(LLM_CONN_MAX_RETRY=-1)

    with pytest.raises(ValidationError):
        LLMSettings(LLM_CONN_RETRY_INTERVAL_SEC=-0.1)

    with pytest.raises(ValidationError):
        LLMSettings(TURN_LLM_STALL_TIMEOUT_SEC=0.0)

    with pytest.raises(ValidationError):
        LLMSettings(MCP_STARTUP_GREETING_TIMEOUT_SEC=-0.1)


def test_voice_runtime_tuning_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        VoiceSettings(POCKET_TTS_CONN_TIMEOUT_SEC=0.0)

    with pytest.raises(
        ValidationError,
        match="TTS_PROVIDER must be either 'pocket', 'deepgram', or 'nvidia'",
    ):
        VoiceSettings(TTS_PROVIDER="invalid")

def test_voice_runtime_tuning_accepts_deepgram_provider_with_key() -> None:
    settings = VoiceSettings(TTS_PROVIDER="DeepGram", DEEPGRAM_API_KEY="test-key")

    assert settings.TTS_PROVIDER == "deepgram"
    assert settings.DEEPGRAM_API_KEY == "test-key"


def test_voice_runtime_tuning_accepts_nvidia_provider() -> None:
    settings = VoiceSettings(TTS_PROVIDER="NVIDIA")

    assert settings.TTS_PROVIDER == "nvidia"


def test_stt_runtime_tuning_accepts_deepgram_provider() -> None:
    settings = STTSettings(STT_PROVIDER="DeepGram")

    assert settings.STT_PROVIDER == "deepgram"


def test_stt_runtime_tuning_validation_rejects_invalid_provider() -> None:
    with pytest.raises(
        ValidationError,
        match="STT_PROVIDER must be either 'moonshine', 'nvidia', or 'deepgram'",
    ):
        STTSettings(STT_PROVIDER="invalid")


def test_settings_require_shared_deepgram_key_for_tts_provider() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "DEEPGRAM_API_KEY is required when TTS_PROVIDER=deepgram "
            "or STT_PROVIDER=deepgram"
        ),
    ):
        Settings(
            voice=VoiceSettings(TTS_PROVIDER="deepgram", DEEPGRAM_API_KEY=" "),
            llm=LLMSettings(OLLAMA_API_KEY="test-key"),
        )


def test_settings_require_shared_deepgram_key_for_stt_provider() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "DEEPGRAM_API_KEY is required when TTS_PROVIDER=deepgram "
            "or STT_PROVIDER=deepgram"
        ),
    ):
        Settings(
            voice=VoiceSettings(TTS_PROVIDER="pocket", DEEPGRAM_API_KEY=" "),
            stt=STTSettings(STT_PROVIDER="deepgram"),
            llm=LLMSettings(OLLAMA_API_KEY="test-key"),
        )


def test_settings_allow_shared_deepgram_key_for_stt_provider() -> None:
    settings = Settings(
        voice=VoiceSettings(TTS_PROVIDER="pocket", DEEPGRAM_API_KEY="deepgram-key"),
        stt=STTSettings(STT_PROVIDER="deepgram"),
        llm=LLMSettings(OLLAMA_API_KEY="test-key"),
    )

    assert settings.stt.STT_PROVIDER == "deepgram"
    assert settings.voice.DEEPGRAM_API_KEY == "deepgram-key"


def test_settings_require_nvidia_tts_or_shared_key_when_ssl_enabled() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "NVIDIA_TTS_API_KEY or NVIDIA_API_KEY is required when "
            "TTS_PROVIDER=nvidia and NVIDIA_TTS_USE_SSL=true"
        ),
    ):
        Settings(
            voice=VoiceSettings(TTS_PROVIDER="nvidia", NVIDIA_TTS_USE_SSL=True),
            llm=LLMSettings(OLLAMA_API_KEY="test-key", NVIDIA_API_KEY=None),
        )


def test_settings_allow_nvidia_tts_without_key_when_ssl_disabled() -> None:
    settings = Settings(
        voice=VoiceSettings(TTS_PROVIDER="nvidia", NVIDIA_TTS_USE_SSL=False),
        llm=LLMSettings(OLLAMA_API_KEY="test-key", NVIDIA_API_KEY=None),
    )

    assert settings.voice.TTS_PROVIDER == "nvidia"
    assert settings.voice.NVIDIA_TTS_USE_SSL is False


def test_settings_allow_nvidia_tts_with_shared_nvidia_api_key() -> None:
    settings = Settings(
        voice=VoiceSettings(TTS_PROVIDER="nvidia", NVIDIA_TTS_USE_SSL=True),
        llm=LLMSettings(OLLAMA_API_KEY="test-key", NVIDIA_API_KEY="shared-key"),
    )

    assert settings.voice.TTS_PROVIDER == "nvidia"
    assert settings.llm.NVIDIA_API_KEY == "shared-key"


def test_livekit_runtime_tuning_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        LiveKitSettings(LIVEKIT_INITIALIZE_PROCESS_TIMEOUT_SEC=0.0)
