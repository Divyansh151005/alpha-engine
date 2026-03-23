"""Base interface for independent alpha strategy modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.state_engine.state_models import MarketStateVector


class BaseAlphaStrategy(ABC):
    """Abstract alpha strategy contract."""

    @abstractmethod
    def on_market_state(self, state: MarketStateVector) -> None:
        """Optional hook to update internal state from market state."""

    @abstractmethod
    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        """Generate a trading signal for a given market state."""
