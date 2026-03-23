"""Breakout alpha strategy - compression + volatility ONLY."""

from __future__ import annotations

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.state_engine.state_models import MarketStateVector

COMPRESSION_HIGH_THRESHOLD = 0.6
VOLATILITY_EXPANDING_THRESHOLD = 0.5


class BreakoutAlphaStrategy(BaseAlphaStrategy):
    """Breakout: compression + volatility. Direction from price momentum (not EMA)."""

    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        # ONLY compression + volatility for trigger (no trend)
        compression_high = state.compression_state >= COMPRESSION_HIGH_THRESHOLD
        vol_expanding = state.volatility_state >= VOLATILITY_EXPANDING_THRESHOLD

        if not (compression_high and vol_expanding):
            return None

        # Direction: price breakout (momentum = raw price return), NOT trend/EMA
        direction = "long" if state.momentum_pressure > 0.5 else "short"

        # Strength: volatility_state, Confidence: compression_state
        strength = max(0.01, min(1.0, state.volatility_state))
        confidence = max(0.01, min(1.0, state.compression_state))

        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self.__class__.__name__,
            direction=direction,
            strength=strength,
            confidence=confidence,
        )
