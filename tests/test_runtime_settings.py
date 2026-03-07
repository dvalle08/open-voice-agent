from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.settings import LLMSettings, LiveKitSettings, VoiceSettings


def test_llm_runtime_tuning_defaults_are_declared() -> None:
    fields = LLMSettings.model_fields
    settings = LLMSettings(OLLAMA_API_KEY="test-key")

    assert fields["LLM_PROVIDER"].default == "ollama"
    assert fields["MCP_ENABLED"].default is True
    assert fields["MCP_SERVER_URL"].default == "https://huggingface.co/mcp"
    assert fields["MCP_EXTRA_SERVER_URLS"].default == "https://docs.livekit.io/mcp"
    assert fields["OLLAMA_CLOUD_MODE"].default is True
    assert fields["OLLAMA_MODEL"].default == "ministral-3:14b-cloud"
    assert fields["OLLAMA_API_KEY"].default == "ollama"
    assert fields["LLM_CONN_TIMEOUT_SEC"].default == 20.0
    assert fields["LLM_CONN_MAX_RETRY"].default == 1
    assert fields["LLM_CONN_RETRY_INTERVAL_SEC"].default == 1.0
    assert fields["TURN_LLM_STALL_TIMEOUT_SEC"].default == 12.0
    assert fields["MCP_STARTUP_GREETING_TIMEOUT_SEC"].default == 0.0
    assert settings.OLLAMA_BASE_URL == "https://ollama.com/v1"


def test_livekit_runtime_tuning_defaults_are_declared() -> None:
    fields = LiveKitSettings.model_fields

    assert fields["LIVEKIT_NUM_IDLE_PROCESSES"].default == 1
    assert fields["LIVEKIT_INITIALIZE_PROCESS_TIMEOUT_SEC"].default == 20.0
    assert fields["LIVEKIT_JOB_MEMORY_WARN_MB"].default == 6144


def test_voice_runtime_tuning_defaults_are_declared() -> None:
    fields = VoiceSettings.model_fields

    assert fields["POCKET_TTS_CONN_TIMEOUT_SEC"].default == 45.0


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


def test_livekit_runtime_tuning_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        LiveKitSettings(LIVEKIT_INITIALIZE_PROCESS_TIMEOUT_SEC=0.0)
