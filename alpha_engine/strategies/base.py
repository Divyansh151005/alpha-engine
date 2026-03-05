"""Strategy interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStrategy(ABC):
    """Abstract strategy contract."""

    @abstractmethod
    def on_data(self, data: Any) -> None:
        """Receive market data updates."""

    @abstractmethod
    def generate_signal(self) -> Any:
        """Produce signal output for downstream components."""

