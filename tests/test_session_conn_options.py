from __future__ import annotations

from src.agent.runtime import session as runtime_session
from src.core.settings import settings


def test_build_session_connect_options_uses_llm_settings_for_tts_and_llm(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings.llm, "LLM_CONN_TIMEOUT_SEC", 42.5)
    monkeypatch.setattr(settings.llm, "LLM_CONN_MAX_RETRY", 4)
    monkeypatch.setattr(settings.llm, "LLM_CONN_RETRY_INTERVAL_SEC", 0.25)

    llm_conn_options, session_conn_options = runtime_session._build_session_connect_options()

    assert llm_conn_options.timeout == 42.5
    assert llm_conn_options.max_retry == 4
    assert llm_conn_options.retry_interval == 0.25
    assert session_conn_options.llm_conn_options.timeout == 42.5
    assert session_conn_options.llm_conn_options.max_retry == 4
    assert session_conn_options.llm_conn_options.retry_interval == 0.25
    assert session_conn_options.tts_conn_options.timeout == 42.5
    assert session_conn_options.tts_conn_options.max_retry == 4
    assert session_conn_options.tts_conn_options.retry_interval == 0.25
    assert session_conn_options.tts_conn_options is not session_conn_options.llm_conn_options
