"""Allocation interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from alpha_engine.core.state import GlobalState


class BaseAllocator(ABC):
    """Abstract allocator contract."""

    @abstractmethod
    def allocate(self, signals: Any, state: GlobalState) -> Any:
        """Transform signals into target allocations."""

