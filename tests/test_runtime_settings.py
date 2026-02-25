from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.settings import LLMSettings


def test_llm_runtime_tuning_defaults_are_declared() -> None:
    fields = LLMSettings.model_fields

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
