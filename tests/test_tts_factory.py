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
