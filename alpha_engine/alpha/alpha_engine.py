"""Alpha engine orchestrating strategy signal generation."""

from __future__ import annotations

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.core.event_bus import EventBus
from alpha_engine.state_engine.state_models import MarketStateVector


class AlphaEngine:
    """Run all alpha strategies on each market state event."""

    def __init__(self, event_bus: EventBus, strategies: list[BaseAlphaStrategy]) -> None:
        self._event_bus = event_bus
        self._strategies = strategies
        self._event_bus.subscribe("market.state", self._handle_market_state)

    def _handle_market_state(self, state: MarketStateVector) -> None:
        for strategy in self._strategies:
            strategy.on_market_state(state)
            signal = strategy.generate_signal(state)
            if signal is not None:
                self._event_bus.publish("alpha.signal", signal)


class TrendAlphaStrategy(BaseAlphaStrategy):
    """Trend-following strategy over normalized state features."""

    LONG_TREND_THRESHOLD = 0.7
    SHORT_TREND_THRESHOLD = 0.3
    MOMENTUM_THRESHOLD = 0.6

    def on_market_state(self, state: MarketStateVector) -> None:
        # Strategy is stateless for now; hook exists for future extensions.
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        if state.trend_state > self.LONG_TREND_THRESHOLD and state.momentum_pressure > self.MOMENTUM_THRESHOLD:
            direction = "long"
        elif state.trend_state < self.SHORT_TREND_THRESHOLD and state.momentum_pressure > self.MOMENTUM_THRESHOLD:
            direction = "short"
        else:
            return None

        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self.__class__.__name__,
            direction=direction,
            strength=state.trend_state,
            confidence=state.momentum_pressure,
        )
