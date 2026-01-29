import logging
import os
import sys
from typing import Optional

log_level_value = os.getenv("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_value.upper(), logging.INFO)

log_format = "%(asctime)s - %(levelname)s - %(name)s - [%(process)d] %(message)s"

logging.basicConfig(
    level=log_level,
    format=log_format,
    stream=sys.stdout,
    force=True,
)

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
