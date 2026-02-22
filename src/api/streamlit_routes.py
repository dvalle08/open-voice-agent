from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict
from http import HTTPStatus
from typing import Any, Callable

from streamlit import config
from streamlit.web.server.server_util import make_url_path_regex
from tornado.routing import PathMatches, Rule
from tornado.web import Application, RequestHandler

from src.api.session_bootstrap import (
    build_bootstrap_error_payload,
    build_session_bootstrap_payload,
)
from src.core.logger import logger

SESSION_BOOTSTRAP_ENDPOINT = "session/bootstrap"

_patch_lock = threading.Lock()
_patch_installed = False
_original_create_app: Callable[..., Application] | None = None


def build_session_bootstrap_path() -> str:
    base_path = (config.get_option("server.baseUrlPath") or "").strip("/")
    if not base_path:
        return "/session/bootstrap"
    return f"/{base_path}/session/bootstrap"


class SessionBootstrapRequestHandler(RequestHandler):
    def set_default_headers(self) -> None:
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
        self.set_header("Cache-Control", "no-store")

    def options(self) -> None:
        self.set_status(HTTPStatus.NO_CONTENT)
        self.finish()

    async def get(self) -> None:
        try:
            payload = await asyncio.to_thread(build_session_bootstrap_payload)
        except Exception as exc:  # pragma: no cover - integration behavior
            logger.exception("Failed to create session bootstrap payload: %s", exc)
            self._write_json(
                build_bootstrap_error_payload(exc),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._write_json(asdict(payload), status=HTTPStatus.OK)

    def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus) -> None:
        self.set_status(status)
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.finish(json.dumps(payload))


def install_streamlit_bootstrap_route() -> None:
    global _patch_installed, _original_create_app

    if _patch_installed:
        return

    with _patch_lock:
        if _patch_installed:
            return

        from streamlit.web.server import server as server_module

        _original_create_app = server_module.Server._create_app

        def _patched_create_app(self: Any) -> Application:
            app = _original_create_app(self)
            _install_route_on_app(app)
            return app

        server_module.Server._create_app = _patched_create_app
        _patch_installed = True
        logger.info(
            "Installed Streamlit bootstrap route at %s",
            build_session_bootstrap_path(),
        )


def _install_route_on_app(app: Application) -> None:
    if getattr(app, "_open_voice_bootstrap_route_installed", False):
        return

    base = config.get_option("server.baseUrlPath")
    pattern = make_url_path_regex(base, SESSION_BOOTSTRAP_ENDPOINT)
    # Ensure this route is checked before Streamlit's static catch-all route.
    app.wildcard_router.rules.insert(
        0,
        Rule(PathMatches(pattern), SessionBootstrapRequestHandler),
    )

    setattr(app, "_open_voice_bootstrap_route_installed", True)
