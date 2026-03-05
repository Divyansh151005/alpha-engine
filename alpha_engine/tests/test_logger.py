"""Tests for logger utility."""

from __future__ import annotations

import logging

from alpha_engine.core.logger import get_logger


def test_get_logger_returns_logger_instance() -> None:
    logger = get_logger("alpha_engine.tests")
    assert isinstance(logger, logging.Logger)


def test_logger_logs_without_crashing(caplog) -> None:
    logger = get_logger("alpha_engine.tests.logger")
    with caplog.at_level(logging.INFO):
        logger.info("log smoke test")

