"""Risk manager interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from alpha_engine.core.state import GlobalState


class BaseRiskManager(ABC):
    """Abstract risk manager contract."""

    @abstractmethod
    def apply(self, position_size: float, state: GlobalState) -> float:
        """Adjust position size according to risk constraints."""

