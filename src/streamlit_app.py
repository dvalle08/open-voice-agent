from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.api.session_bootstrap import ensure_session_bootstrap_server
from src.core.settings import settings

UI_DIR = Path(__file__).parent / "ui"
INDEX_TEMPLATE = UI_DIR / "index.html"
MAIN_JS = UI_DIR / "main.js"


def extract_display_name(model_id: str, provider: str) -> str:
    """Extract human-readable display name from model ID."""
    # Get the model name (part after '/' if present)
    if "/" in model_id:
        name = model_id.split("/")[-1]
    else:
        name = model_id

    # Special cases for well-known models
    if "moonshine" in name.lower():
        return "Moonshine STT"
    elif "qwen" in name.lower():
        return name  # Keep format like "Qwen2.5-3B-Instruct"
    elif "parakeet" in name.lower():
        return "NVIDIA Parakeet"
    elif "llama" in name.lower():
        parts = name.split("-")
        return f"Llama {parts[1]}" if len(parts) > 1 else name

    # Default: basic cleanup
    return name.replace("-", " ").title()


def generate_footer_html() -> str:
    """Generate dynamic 'Powered by' footer HTML based on current settings."""
    # STT Component
    stt_provider = settings.stt.STT_PROVIDER.lower()
    if stt_provider == "moonshine":
        stt_model_id = settings.stt.MOONSHINE_MODEL_ID
        stt_url = f"https://huggingface.co/{stt_model_id}"
        stt_display = extract_display_name(stt_model_id, "moonshine")
    elif stt_provider == "nvidia":
        stt_url = "https://build.nvidia.com/"
        stt_display = extract_display_name(settings.stt.NVIDIA_STT_MODEL, "nvidia")
    else:
        stt_url = "#"
        stt_display = f"Unknown STT ({stt_provider})"

    # LLM Component
    llm_provider = settings.llm.LLM_PROVIDER.lower()
    if llm_provider == "huggingface":
        llm_model_id = settings.llm.HUGGINGFACE_MODEL_ID
        llm_url = f"https://huggingface.co/{llm_model_id}"
        llm_display = extract_display_name(llm_model_id, "huggingface")
    elif llm_provider == "nvidia":
        llm_url = "https://build.nvidia.com/"
        llm_display = extract_display_name(settings.llm.NVIDIA_MODEL, "nvidia")
    else:
        llm_url = "#"
        llm_display = f"Unknown LLM ({llm_provider})"

    # Build footer HTML
    return f"""Powered by open-source core components:
        <a href="{stt_url}" target="_blank" rel="noopener noreferrer">{stt_display}</a> ·
        <a href="{llm_url}" target="_blank" rel="noopener noreferrer">{llm_display}</a> ·
        <a href="https://huggingface.co/kyutai/pocket-tts" target="_blank" rel="noopener noreferrer">PocketTTS</a> ·
        <a href="https://livekit.io/" target="_blank" rel="noopener noreferrer">LiveKit</a>"""


def render_client(*, livekit_url: str, session_bootstrap_url: str) -> None:
    template = INDEX_TEMPLATE.read_text(encoding="utf-8")
    js = MAIN_JS.read_text(encoding="utf-8")
    footer_html = generate_footer_html()

    html = (
        template.replace("{{MAIN_JS}}", js)
        .replace("{{LIVEKIT_URL_JSON}}", json.dumps(livekit_url))
        .replace("{{SESSION_BOOTSTRAP_URL_JSON}}", json.dumps(session_bootstrap_url))
        .replace("{{FOOTER_POWERED_BY}}", footer_html)
    )

    st.components.v1.html(html, height=900, scrolling=False)


def main() -> None:
    st.set_page_config(page_title="Open Voice Agent", layout="wide")
    st.markdown(
        "<style>"
        "header {visibility: hidden;} "
        ".stMainBlockContainer {padding-top: 1rem; padding-bottom: 0;} "
        "iframe {border: 1px solid #252d3f; border-radius: 12px;}"
        "</style>",
        unsafe_allow_html=True,
    )

    if not settings.livekit.LIVEKIT_URL:
        st.error("LIVEKIT_URL is not set in the environment.")
        st.stop()
    if not settings.livekit.LIVEKIT_API_KEY or not settings.livekit.LIVEKIT_API_SECRET:
        st.error("LIVEKIT_API_KEY or LIVEKIT_API_SECRET is not set.")
        st.stop()

    try:
        session_bootstrap_url = ensure_session_bootstrap_server()
    except Exception as exc:
        st.error(f"Failed to start session bootstrap service: {exc}")
        st.stop()

    render_client(
        livekit_url=settings.livekit.LIVEKIT_URL,
        session_bootstrap_url=session_bootstrap_url,
    )


if __name__ == "__main__":
    main()
