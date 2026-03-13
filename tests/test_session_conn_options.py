from __future__ import annotations

import asyncio
import types
from pathlib import Path

from livekit.agents.inference_runner import _InferenceRunner

from src.agent.runtime import session as runtime_session
from src.core.settings import VoiceSettings, settings


ENV_EXAMPLE_PATH = Path(__file__).resolve().parents[1] / ".env.example"


class _FakeJobContext:
    def __init__(self) -> None:
        self.room = types.SimpleNamespace(name="room-name")
        self.job = types.SimpleNamespace(
            id="job-id",
            participant=types.SimpleNamespace(identity="participant-id"),
            room=types.SimpleNamespace(sid="room-sid"),
            metadata="",
        )
        self.shutdown_callbacks: list[object] = []

    def add_shutdown_callback(self, callback) -> None:
        self.shutdown_callbacks.append(callback)


class _FakeToolFeedbackController:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.start_calls: list[dict[str, object]] = []

    async def start(self, *, room, session) -> None:
        self.start_calls.append({"room": room, "session": session})

    async def aclose(self) -> None:
        return None


def test_build_session_connect_options_uses_separate_tts_timeout_and_shared_retry_settings(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings.llm, "LLM_CONN_TIMEOUT_SEC", 42.5)
    monkeypatch.setattr(settings.llm, "LLM_CONN_MAX_RETRY", 4)
    monkeypatch.setattr(settings.llm, "LLM_CONN_RETRY_INTERVAL_SEC", 0.25)
    monkeypatch.setattr(settings.voice, "POCKET_TTS_CONN_TIMEOUT_SEC", 45.0)

    llm_conn_options, session_conn_options = runtime_session._build_session_connect_options()

    assert llm_conn_options.timeout == 42.5
    assert llm_conn_options.max_retry == 4
    assert llm_conn_options.retry_interval == 0.25
    assert session_conn_options.llm_conn_options.timeout == 42.5
    assert session_conn_options.llm_conn_options.max_retry == 4
    assert session_conn_options.llm_conn_options.retry_interval == 0.25
    assert session_conn_options.tts_conn_options.timeout == 45.0
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


def test_importing_session_registers_english_turn_detector_runner() -> None:
    assert "lk_end_of_utterance_en" in _InferenceRunner.registered_runners


def test_resolve_stt_metrics_model_name_uses_deepgram_model(monkeypatch) -> None:
    monkeypatch.setattr(settings.stt, "STT_PROVIDER", "deepgram")
    monkeypatch.setattr(settings.stt, "DEEPGRAM_STT_MODEL", "nova-3")

    model_name = runtime_session._resolve_stt_metrics_model_name()

    assert model_name == "nova-3"


def test_session_handler_runs_llm_warmup_before_session_start(monkeypatch) -> None:
    order: list[str] = []
    created_sessions: list[object] = []
    ctx = _FakeJobContext()

    class _FakeAgentSession:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.start_calls: list[dict[str, object]] = []
            created_sessions.append(self)

        async def start(self, **kwargs) -> None:
            order.append("start")
            self.start_calls.append(kwargs)

    monkeypatch.setattr(runtime_session, "setup_langfuse_tracer", lambda: None)
    monkeypatch.setattr(runtime_session, "MetricsCollector", lambda **kwargs: object())
    monkeypatch.setattr(runtime_session, "create_tts", lambda: object())
    monkeypatch.setattr(
        runtime_session,
        "build_llm_runtime",
        lambda _settings: types.SimpleNamespace(
            llm=object(),
            provider="ollama",
            model="test-model",
            mcp_runtime_active=False,
            mcp_servers=None,
        ),
    )
    monkeypatch.setattr(runtime_session, "create_stt", lambda: "stt")
    monkeypatch.setattr(runtime_session, "ToolFeedbackController", _FakeToolFeedbackController)
    monkeypatch.setattr(runtime_session, "AgentSession", _FakeAgentSession)
    monkeypatch.setattr(runtime_session, "Assistant", lambda **kwargs: "assistant")
    monkeypatch.setattr(runtime_session, "install_mcp_generate_reply_guard", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_session, "run_startup_greeting", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_session.silero.VAD, "load", lambda **kwargs: "vad")
    monkeypatch.setattr(runtime_session, "EnglishModel", lambda: "turn-detector")
    monkeypatch.setattr(runtime_session.room_io, "AudioInputOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(runtime_session.room_io, "RoomOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(settings.voice, "MIN_ENDPOINTING_DELAY", 1.0)
    monkeypatch.setattr(settings.voice, "MAX_ENDPOINTING_DELAY", 4.0)
    monkeypatch.setattr(settings.voice, "PREEMPTIVE_GENERATION", False)

    async def _fake_run_llm_warmup(**kwargs) -> None:
        order.append("llm")

    monkeypatch.setattr(runtime_session, "run_llm_warmup", _fake_run_llm_warmup)

    asyncio.run(runtime_session.session_handler(ctx))

    assert order == ["llm", "start"]
    assert len(created_sessions) == 1
    assert created_sessions[0].start_calls
    assert created_sessions[0].kwargs["turn_detection"] == "turn-detector"
    assert created_sessions[0].kwargs["min_endpointing_delay"] == 1.0
    assert created_sessions[0].kwargs["max_endpointing_delay"] == 4.0
    assert created_sessions[0].kwargs["preemptive_generation"] is False


def test_env_example_turn_profile_matches_voice_defaults() -> None:
    env_values = _parse_env_file(ENV_EXAMPLE_PATH)

    assert env_values["LIVEKIT_FRAME_SIZE_MS"] == str(
        VoiceSettings.model_fields["LIVEKIT_FRAME_SIZE_MS"].default
    )
    assert env_values["VAD_MIN_SILENCE_DURATION"] == str(
        VoiceSettings.model_fields["VAD_MIN_SILENCE_DURATION"].default
    )
    assert env_values["VAD_THRESHOLD"] == str(
        VoiceSettings.model_fields["VAD_THRESHOLD"].default
    )
    assert env_values["MIN_ENDPOINTING_DELAY"] == str(
        VoiceSettings.model_fields["MIN_ENDPOINTING_DELAY"].default
    )
    assert env_values["MAX_ENDPOINTING_DELAY"] == str(
        VoiceSettings.model_fields["MAX_ENDPOINTING_DELAY"].default
    )
    assert env_values["PREEMPTIVE_GENERATION"] == _env_bool(
        VoiceSettings.model_fields["PREEMPTIVE_GENERATION"].default
    )


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.split(" #", 1)[0].strip()
    return values


def _env_bool(value: bool) -> str:
    return "true" if value else "false"
