from __future__ import annotations

from src.agent.models import tts_factory
from src.core.settings import settings


def test_create_tts_uses_pocket_provider_settings(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class _FakePocketTTS:
        def __init__(self, *, voice: str, temperature: float, lsd_decode_steps: int) -> None:
            calls["voice"] = voice
            calls["temperature"] = temperature
            calls["lsd_decode_steps"] = lsd_decode_steps

    monkeypatch.setattr(settings.voice, "TTS_PROVIDER", "pocket")
    monkeypatch.setattr(settings.voice, "POCKET_TTS_VOICE", "alba")
    monkeypatch.setattr(settings.voice, "POCKET_TTS_TEMPERATURE", 0.4)
    monkeypatch.setattr(settings.voice, "POCKET_TTS_LSD_DECODE_STEPS", 2)
    monkeypatch.setattr(tts_factory, "PocketTTS", _FakePocketTTS)

    tts_engine = tts_factory.create_tts()

    assert isinstance(tts_engine, _FakePocketTTS)
    assert calls == {
        "voice": "alba",
        "temperature": 0.4,
        "lsd_decode_steps": 2,
    }


def test_create_tts_uses_deepgram_provider(monkeypatch) -> None:
    deepgram_calls: dict[str, object] = {}

    class _FakeDeepgramTTS:
        def __init__(self, *, model: str, api_key: str | None = None) -> None:
            deepgram_calls["model"] = model
            deepgram_calls["api_key"] = api_key

    monkeypatch.setattr(settings.voice, "TTS_PROVIDER", "deepgram")
    monkeypatch.setattr(settings.voice, "DEEPGRAM_API_KEY", "deepgram-test-key")
    monkeypatch.setattr(tts_factory.deepgram, "TTS", _FakeDeepgramTTS)

    tts_engine = tts_factory.create_tts()

    assert isinstance(tts_engine, _FakeDeepgramTTS)
    assert deepgram_calls == {
        "model": "aura-2-thalia-en",
        "api_key": "deepgram-test-key",
    }


def test_create_tts_uses_nvidia_provider_with_shared_api_key_fallback(monkeypatch) -> None:
    nvidia_calls: dict[str, object] = {}
    patch_calls: list[str] = []

    class _FakeNvidiaTTS:
        def __init__(
            self,
            *,
            voice: str,
            language_code: str,
            server: str,
            function_id: str,
            use_ssl: bool,
            api_key: str | None = None,
        ) -> None:
            nvidia_calls["voice"] = voice
            nvidia_calls["language_code"] = language_code
            nvidia_calls["server"] = server
            nvidia_calls["function_id"] = function_id
            nvidia_calls["use_ssl"] = use_ssl
            nvidia_calls["api_key"] = api_key

    monkeypatch.setattr(settings.voice, "TTS_PROVIDER", "nvidia")
    monkeypatch.setattr(settings.voice, "NVIDIA_TTS_API_KEY", None)
    monkeypatch.setattr(settings.voice, "NVIDIA_TTS_VOICE", "Magpie-Multilingual.EN-US.Leo")
    monkeypatch.setattr(settings.voice, "NVIDIA_TTS_LANGUAGE_CODE", "en-US")
    monkeypatch.setattr(settings.voice, "NVIDIA_TTS_SERVER", "grpc.nvcf.nvidia.com:443")
    monkeypatch.setattr(
        settings.voice,
        "NVIDIA_TTS_FUNCTION_ID",
        "877104f7-e885-42b9-8de8-f6e4c6303969",
    )
    monkeypatch.setattr(settings.voice, "NVIDIA_TTS_USE_SSL", True)
    monkeypatch.setattr(settings.llm, "NVIDIA_API_KEY", "shared-nvidia-test-key")
    monkeypatch.setattr(
        tts_factory,
        "_patch_nvidia_tts_stream_once",
        lambda: patch_calls.append("patched"),
    )
    monkeypatch.setattr(tts_factory.nvidia, "TTS", _FakeNvidiaTTS)

    tts_engine = tts_factory.create_tts()

    assert isinstance(tts_engine, _FakeNvidiaTTS)
    assert patch_calls == ["patched"]
    assert nvidia_calls == {
        "voice": "Magpie-Multilingual.EN-US.Leo",
        "language_code": "en-US",
        "server": "grpc.nvcf.nvidia.com:443",
        "function_id": "877104f7-e885-42b9-8de8-f6e4c6303969",
        "use_ssl": True,
        "api_key": "shared-nvidia-test-key",
    }
