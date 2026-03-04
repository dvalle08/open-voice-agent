"""Error helper utilities for session logging."""

from typing import Any


def error_type_name(error_obj: Any) -> str:
    return getattr(error_obj, "type", type(error_obj).__name__)


def error_recoverable(error_obj: Any) -> str:
    recoverable = getattr(error_obj, "recoverable", None)
    if recoverable is None:
        return "unknown"
    return str(bool(recoverable)).lower()


def error_detail(error_obj: Any) -> str:
    nested_error = getattr(error_obj, "error", None)
    if nested_error:
        return str(nested_error)
    return str(error_obj)
