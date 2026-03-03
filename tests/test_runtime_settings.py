from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.settings import LLMSettings


def test_llm_runtime_tuning_defaults_are_declared() -> None:
    fields = LLMSettings.model_fields

    assert fields["LLM_PROVIDER"].default == "ollama"
    assert fields["MCP_ENABLED"].default is True
    assert fields["MCP_SERVER_URL"].default == "https://huggingface.co/mcp"
    assert fields["OLLAMA_BASE_URL"].default == "http://localhost:11434/v1"
    assert fields["OLLAMA_MODEL"].default == "qwen2.5:7b"
    assert fields["OLLAMA_API_KEY"].default == "ollama"
    assert fields["LLM_CONN_TIMEOUT_SEC"].default == 12.0
    assert fields["LLM_CONN_MAX_RETRY"].default == 1
    assert fields["LLM_CONN_RETRY_INTERVAL_SEC"].default == 1.0
    assert fields["TURN_LLM_STALL_TIMEOUT_SEC"].default == 8.0


def test_llm_runtime_tuning_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        LLMSettings(LLM_CONN_TIMEOUT_SEC=0.0)

    with pytest.raises(ValidationError):
        LLMSettings(LLM_CONN_MAX_RETRY=-1)

    with pytest.raises(ValidationError):
        LLMSettings(LLM_CONN_RETRY_INTERVAL_SEC=-0.1)

    with pytest.raises(ValidationError):
        LLMSettings(TURN_LLM_STALL_TIMEOUT_SEC=0.0)
