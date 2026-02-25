import logging
import os
import sys
from typing import Optional

log_level_value = os.getenv("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_value.upper(), logging.INFO)

log_format = "%(asctime)s - %(levelname)s - %(name)s - [%(process)d] %(message)s"

_default_root_handler: logging.Handler | None = None


def _is_agent_cli_invocation() -> bool:
    script = os.path.basename(sys.argv[0]).lower()
    if script != "agent.py":
        return False
    return any(arg in {"start", "console", "download-files", "dev"} for arg in sys.argv[1:])


def _configure_default_root_handler() -> None:
    """Configure root logging only when no handler is already installed."""
    global _default_root_handler

    if _is_agent_cli_invocation():
        return

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(log_level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    root.addHandler(handler)
    root.setLevel(log_level)
    _default_root_handler = handler


def detach_default_root_handler() -> None:
    """Detach the handler installed by this module, if present."""
    global _default_root_handler

    if _default_root_handler is None:
        return

    root = logging.getLogger()
    if _default_root_handler in root.handlers:
        root.removeHandler(_default_root_handler)
    _default_root_handler = None


_configure_default_root_handler()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("torchaudio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger_name = name or "open_voice_agent"
    return logging.getLogger(logger_name)


logger = get_logger()
