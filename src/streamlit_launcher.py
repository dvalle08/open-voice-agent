from __future__ import annotations

import os

from streamlit.web import bootstrap

from src.api.streamlit_routes import install_streamlit_bootstrap_route


def main() -> None:
    os.environ.setdefault("OPEN_VOICE_USE_STREAMLIT_BOOTSTRAP_ROUTE", "1")
    install_streamlit_bootstrap_route()

    port = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
    address = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

    bootstrap.run(
        main_script_path="src/streamlit_app.py",
        is_hello=False,
        args=[],
        flag_options={
            "server.port": port,
            "server.address": address,
            "server.useStarlette": False,
        },
    )


if __name__ == "__main__":
    main()
