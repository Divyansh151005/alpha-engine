"""Execution engine interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseExecutionEngine(ABC):
    """Abstract execution engine contract."""

    @abstractmethod
    def execute(self, order: Any) -> Any:
        """Execute an order request."""

