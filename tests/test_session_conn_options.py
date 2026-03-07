from __future__ import annotations

from livekit.agents.inference_runner import _InferenceRunner

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


def test_build_server_uses_livekit_process_initialization_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings.livekit, "LIVEKIT_NUM_IDLE_PROCESSES", 3)
    monkeypatch.setattr(settings.livekit, "LIVEKIT_INITIALIZE_PROCESS_TIMEOUT_SEC", 30.0)
    monkeypatch.setattr(settings.livekit, "LIVEKIT_JOB_MEMORY_WARN_MB", 8192.0)

    server = runtime_session._build_server()

    assert server._num_idle_processes == 3
    assert server._initialize_process_timeout == 30.0
    assert server._job_memory_warn_mb == 8192.0


def test_importing_session_registers_multilingual_turn_detector_runner() -> None:
    assert "lk_end_of_utterance_multilingual" in _InferenceRunner.registered_runners
