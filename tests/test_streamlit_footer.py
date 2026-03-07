from __future__ import annotations

import pytest

from src import streamlit_app


def test_generate_footer_html_uses_ollama_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streamlit_app.settings.llm, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(streamlit_app.settings.llm, "OLLAMA_MODEL", "qwen2.5:7b")
    monkeypatch.setattr(streamlit_app.settings.voice, "TTS_PROVIDER", "pocket")
    monkeypatch.setattr(streamlit_app.settings.voice, "POCKET_TTS_VOICE", "alba")

    html = streamlit_app.generate_footer_html()

    assert "https://ollama.com/" in html
    assert "Ollama (qwen2.5:7b)" in html
    assert "https://huggingface.co/kyutai/pocket-tts" in html
    assert "PocketTTS (voice: alba)" in html
    assert "Deepgram (aura-2-thalia-en)" not in html


def test_generate_footer_html_uses_deepgram_tts_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streamlit_app.settings.voice, "TTS_PROVIDER", "deepgram")

    html = streamlit_app.generate_footer_html()

    assert "https://deepgram.com/" in html
    assert "Deepgram (aura-2-thalia-en)" in html
    assert "PocketTTS (voice:" not in html
