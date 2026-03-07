from __future__ import annotations

from src.agent.models import stt_factory
from src.core.settings import settings


def test_create_stt_uses_deepgram_provider(monkeypatch) -> None:
    deepgram_calls: dict[str, object] = {}

    class _FakeDeepgramSTT:
        def __init__(
            self,
            *,
            model: str,
            language: str,
            api_key: str | None = None,
        ) -> None:
            deepgram_calls["model"] = model
            deepgram_calls["language"] = language
            deepgram_calls["api_key"] = api_key

    monkeypatch.setattr(settings.stt, "STT_PROVIDER", "deepgram")
    monkeypatch.setattr(settings.stt, "DEEPGRAM_STT_MODEL", "nova-3")
    monkeypatch.setattr(settings.stt, "DEEPGRAM_STT_LANGUAGE", "en-US")
    monkeypatch.setattr(settings.voice, "DEEPGRAM_API_KEY", "deepgram-test-key")
    monkeypatch.setattr(stt_factory.deepgram, "STT", _FakeDeepgramSTT)

    stt_engine = stt_factory.create_stt()

    assert isinstance(stt_engine, _FakeDeepgramSTT)
    assert deepgram_calls == {
        "model": "nova-3",
        "language": "en-US",
        "api_key": "deepgram-test-key",
    }
