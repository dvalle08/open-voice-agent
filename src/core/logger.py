"""Centralized logging service for Open Voice Agent.

Provides configurable logging with structured output format.
Log level can be configured via LOG_LEVEL environment variable.
"""

import logging
import os
import sys
from typing import Optional

# Get log level from environment
log_level_value = os.getenv("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_value.upper(), logging.INFO)

# Structured format with timestamp and module info
log_format = "%(asctime)s - %(levelname)s - %(name)s - [%(process)d] %(message)s"

logging.basicConfig(
    level=log_level,
    format=log_format,
    stream=sys.stdout,
    force=True,
)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("torchaudio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Optional logger name. Defaults to 'open_voice_agent'.

    Returns:
        A configured logging.Logger instance.
    """
    logger_name = name or "open_voice_agent"
    return logging.getLogger(logger_name)


# Default logger instance
logger = get_logger()
