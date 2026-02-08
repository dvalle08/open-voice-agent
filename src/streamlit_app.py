from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import streamlit as st

from src.api.livekit_tokens import create_room_token
from src.core.settings import settings

UI_DIR = Path(__file__).parent / "ui"
INDEX_TEMPLATE = UI_DIR / "index.html"
MAIN_JS = UI_DIR / "main.js"


def render_client(token: str, livekit_url: str) -> None:
    template = INDEX_TEMPLATE.read_text(encoding="utf-8")
    js = MAIN_JS.read_text(encoding="utf-8")

    html = (
        template.replace("{{MAIN_JS}}", js)
        .replace("{{LIVEKIT_URL_JSON}}", json.dumps(livekit_url))
        .replace("{{TOKEN_JSON}}", json.dumps(token))
    )

    st.components.v1.html(html, height=360)


def main() -> None:
    st.set_page_config(page_title="Open Voice Agent", layout="centered")
    st.title("Open Voice Agent")
    st.write("Connect to the LiveKit agent and talk using your microphone.")

    if not settings.livekit.LIVEKIT_URL:
        st.error("LIVEKIT_URL is not set in the environment.")
        return
    if not settings.livekit.LIVEKIT_API_KEY or not settings.livekit.LIVEKIT_API_SECRET:
        st.error("LIVEKIT_API_KEY or LIVEKIT_API_SECRET is not set.")
        return

    default_room = f"voice-{uuid4().hex[:8]}"
    room_name = st.text_input("Room name", value=st.session_state.get("room_name", default_room))
    st.session_state["room_name"] = room_name

    start = st.button("Start session", type="primary")
    if start or "token" not in st.session_state:
        token_data = create_room_token(room_name=room_name)
        st.session_state["token"] = token_data.token

    token = st.session_state.get("token")
    if not token:
        st.info("Click Start session to generate a LiveKit token.")
        return

    render_client(token=token, livekit_url=settings.livekit.LIVEKIT_URL)


if __name__ == "__main__":
    main()
