from __future__ import annotations

import pytest

from src import streamlit_app


def test_generate_footer_html_uses_ollama_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streamlit_app.settings.llm, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(streamlit_app.settings.llm, "OLLAMA_MODEL", "qwen2.5:7b")

    html = streamlit_app.generate_footer_html()

    assert "https://ollama.com/" in html
    assert "qwen2.5:7b" in html
