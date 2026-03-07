from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import sleep
from typing import Any
from uuid import uuid4

from livekit import api as livekit_api

from src.api.livekit_tokens import create_room_token, ensure_agent_dispatched_sync
from src.core.logger import logger
from src.core.settings import settings

BOOTSTRAP_MAX_ATTEMPTS = 4
BOOTSTRAP_RETRY_DELAYS_SEC = (0.25, 0.5, 1.0)
RETRYABLE_TWIRP_CODES = {
    livekit_api.TwirpErrorCode.ABORTED,
    livekit_api.TwirpErrorCode.DEADLINE_EXCEEDED,
    livekit_api.TwirpErrorCode.INTERNAL,
    livekit_api.TwirpErrorCode.RESOURCE_EXHAUSTED,
    livekit_api.TwirpErrorCode.UNAVAILABLE,
    livekit_api.TwirpErrorCode.UNKNOWN,
}
CLIENT_BOOTSTRAP_ERROR_MESSAGE = "Could not initialize voice session. Please try again."


@dataclass(frozen=True)
class SessionBootstrapPayload:
    room_name: str
    token: str
    participant_identity: str
    session_id: str
    dispatch_id: str | None
    dispatch_worker_id: str | None
    langfuse_enabled: bool
    langfuse_host: str | None
    langfuse_project_id: str | None
    langfuse_public_traces: bool


def build_session_bootstrap_payload() -> SessionBootstrapPayload:
    """Create a brand-new room/token/dispatch payload for a connect attempt."""
    attempt = 1
    while True:
        try:
            return _build_session_bootstrap_payload_once()
        except Exception as exc:
            if attempt >= BOOTSTRAP_MAX_ATTEMPTS or not _is_retryable_bootstrap_error(exc):
                raise

            delay = BOOTSTRAP_RETRY_DELAYS_SEC[min(attempt - 1, len(BOOTSTRAP_RETRY_DELAYS_SEC) - 1)]
            logger.warning(
                "Bootstrap payload attempt %s/%s failed with retryable error (%s): %s; retrying in %.2fs",
                attempt,
                BOOTSTRAP_MAX_ATTEMPTS,
                type(exc).__name__,
                exc,
                delay,
            )
            sleep(delay)
            attempt += 1


def _build_session_bootstrap_payload_once() -> SessionBootstrapPayload:
    """Create one room/token/dispatch payload without retries."""
    room_name = f"voice-{uuid4().hex[:8]}"
    token_data = create_room_token(room_name=room_name)
    session_id = str(uuid4())
    metadata_payload = {
        "type": "session_meta",
        "session_id": session_id,
        "participant_id": token_data.identity,
    }
    dispatch = ensure_agent_dispatched_sync(
        room_name=room_name,
        agent_name=settings.livekit.LIVEKIT_AGENT_NAME,
        metadata=json.dumps(metadata_payload),
        reset_existing=True,
    )

    assigned_worker_id = None
    for job in getattr(dispatch.state, "jobs", []):
        state = getattr(job, "state", None)
        if state and getattr(state, "worker_id", None):
            assigned_worker_id = state.worker_id
            break

    langfuse_host = _normalize_langfuse_host()
    langfuse_project_id = _normalize_langfuse_project_id()

    payload = SessionBootstrapPayload(
        room_name=room_name,
        token=token_data.token,
        participant_identity=token_data.identity,
        session_id=session_id,
        dispatch_id=getattr(dispatch, "id", None),
        dispatch_worker_id=assigned_worker_id,
        langfuse_enabled=bool(
            settings.langfuse.LANGFUSE_ENABLED
            and langfuse_host
            and langfuse_project_id
        ),
        langfuse_host=langfuse_host,
        langfuse_project_id=langfuse_project_id,
        langfuse_public_traces=bool(settings.langfuse.LANGFUSE_PUBLIC_TRACES),
    )
    logger.info(
        "Prepared session bootstrap payload room=%s participant=%s dispatch_id=%s worker_id=%s",
        payload.room_name,
        payload.participant_identity,
        payload.dispatch_id or "none",
        payload.dispatch_worker_id or "unassigned",
    )
    return payload


def _normalize_langfuse_host() -> str | None:
    host = settings.langfuse.LANGFUSE_HOST or settings.langfuse.LANGFUSE_BASE_URL
    if not isinstance(host, str):
        return None
    normalized = host.strip().rstrip("/")
    return normalized or None


def _normalize_langfuse_project_id() -> str | None:
    project_id = settings.langfuse.LANGFUSE_PROJECT_ID
    if not isinstance(project_id, str):
        return None
    normalized = project_id.strip()
    return normalized or None


def _is_retryable_bootstrap_error(exc: Exception) -> bool:
    if isinstance(exc, livekit_api.TwirpError):
        return exc.code in RETRYABLE_TWIRP_CODES
    if isinstance(exc, (TimeoutError, OSError, ConnectionError)):
        return True
    return False


def build_bootstrap_error_payload(_: Exception | None = None) -> dict[str, str]:
    """Build a client-safe error response payload without internal details."""
    return {
        "error": "bootstrap_failed",
        "message": CLIENT_BOOTSTRAP_ERROR_MESSAGE,
    }


class _SessionBootstrapHandler(BaseHTTPRequestHandler):
    server_version = "OpenVoiceAgentBootstrap/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/session/bootstrap":
            self._write_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            payload = build_session_bootstrap_payload()
        except Exception as exc:  # pragma: no cover - exercised by integration flow
            logger.exception("Failed to create session bootstrap payload: %s", exc)
            self._write_json(
                build_bootstrap_error_payload(exc),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._write_json(asdict(payload), status=HTTPStatus.OK)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._write_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        # Route server access logs through project logger to keep output consistent.
        logger.debug("Session bootstrap server: " + format, *args)

    def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._write_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


_server_lock = threading.Lock()
_bootstrap_server: ThreadingHTTPServer | None = None
_bootstrap_thread: threading.Thread | None = None


def ensure_session_bootstrap_server() -> str:
    """Start bootstrap server once and return its URL."""
    global _bootstrap_server, _bootstrap_thread

    with _server_lock:
        if _bootstrap_server is None:
            server = ThreadingHTTPServer(("127.0.0.1", 0), _SessionBootstrapHandler)
            server.daemon_threads = True
            thread = threading.Thread(
                target=server.serve_forever,
                name="session-bootstrap-server",
                daemon=True,
            )
            thread.start()
            _bootstrap_server = server
            _bootstrap_thread = thread
            logger.info(
                "Session bootstrap server started at http://127.0.0.1:%s/session/bootstrap",
                server.server_port,
            )

        if _bootstrap_server is None:
            raise RuntimeError("Session bootstrap server did not initialize")
        return f"http://127.0.0.1:{_bootstrap_server.server_port}/session/bootstrap"
