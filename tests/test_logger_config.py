from __future__ import annotations

import logging
import importlib

logger_module = importlib.import_module("src.core.logger")


def test_default_root_handler_is_not_duplicated() -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    try:
        for handler in list(root.handlers):
            root.removeHandler(handler)

        logger_module._default_root_handler = None
        logger_module._configure_default_root_handler()
        first_count = len(root.handlers)
        logger_module._configure_default_root_handler()
        second_count = len(root.handlers)

        assert first_count == 1
        assert second_count == 1

        logger_module.detach_default_root_handler()
        assert len(root.handlers) == 0
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)
        logger_module._default_root_handler = None
