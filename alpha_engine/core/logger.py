"""Logging configuration utilities."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _resolve_log_level() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Create or return a configured logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(_resolve_log_level())

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)

    logger.propagate = False
    return logger

