from __future__ import annotations

import pytest
from tornado.web import Application, RequestHandler

from src.api import streamlit_routes


class _CatchAllHandler(RequestHandler):
    def get(self, *_: str) -> None:
        self.finish("ok")


def test_build_session_bootstrap_path_without_base_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streamlit_routes.config, "get_option", lambda _: "")
    assert streamlit_routes.build_session_bootstrap_path() == "/session/bootstrap"


def test_build_session_bootstrap_path_with_base_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streamlit_routes.config, "get_option", lambda _: "/my/app/")
    assert streamlit_routes.build_session_bootstrap_path() == "/my/app/session/bootstrap"


def test_install_route_on_app_prepends_before_catch_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(streamlit_routes.config, "get_option", lambda _: "")
    app = Application([(r"/(.*)", _CatchAllHandler)])

    streamlit_routes._install_route_on_app(app)

    bootstrap_targets = [
        rule.target
        for rule in app.wildcard_router.rules
        if rule.target is streamlit_routes.SessionBootstrapRequestHandler
    ]
    assert len(bootstrap_targets) == 1
    assert app.wildcard_router.rules[0].target is streamlit_routes.SessionBootstrapRequestHandler

    # Installing twice should not duplicate the bootstrap handler.
    streamlit_routes._install_route_on_app(app)
    bootstrap_targets_after = [
        rule.target
        for rule in app.wildcard_router.rules
        if rule.target is streamlit_routes.SessionBootstrapRequestHandler
    ]
    assert len(bootstrap_targets_after) == 1

