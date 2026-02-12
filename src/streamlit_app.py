from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import streamlit as st

from src.api.livekit_tokens import create_room_token, dispatch_agent_sync
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

    # Auto-generate room name once
    if "room_name" not in st.session_state:
        st.session_state["room_name"] = f"voice-{uuid4().hex[:8]}"

    room_name = st.session_state["room_name"]

    # Auto-create token once
    if "token" not in st.session_state:
        try:
            token_data = create_room_token(room_name=room_name)
            st.session_state["token"] = token_data.token
            st.session_state["agent_dispatched"] = False
        except Exception as exc:
            st.error(f"Failed to create room token: {exc}")
            st.stop()

    # Auto-dispatch agent once
    if not st.session_state.get("agent_dispatched"):
        try:
            dispatch_agent_sync(
                room_name=room_name,
                agent_name=settings.livekit.LIVEKIT_AGENT_NAME,
            )
            st.session_state["agent_dispatched"] = True
        except Exception as exc:
            st.error(f"Failed to dispatch agent: {exc}")
            st.stop()

    render_client(token=st.session_state["token"], livekit_url=settings.livekit.LIVEKIT_URL)


if __name__ == "__main__":
    main()
