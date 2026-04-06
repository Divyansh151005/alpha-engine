"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _execution_related_test_start(request: pytest.FixtureRequest) -> None:
    """Structured log for tests that exercise backtest execution / replay."""
    path = getattr(request.node, "path", None)
    if path is None:
        return
    if Path(path).name not in ("test_backtest_engine.py", "test_filters.py"):
        return
    print(f"[TEST START] {request.node.name}")
