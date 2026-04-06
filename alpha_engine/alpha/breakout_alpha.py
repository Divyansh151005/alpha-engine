"""Breakout alpha: range expansion, volatility spike, price vs recent high/low. Score-based."""

from __future__ import annotations

import math

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.state_engine.state_models import MarketStateVector

SCORE_NORMALIZE_WINDOW = 50
_EPS = 1e-12


class BreakoutAlphaStrategy(BaseAlphaStrategy):
    """Score-based breakout: range expansion + vol spike + price above/below recent high/low."""

    def __init__(self) -> None:
        self._score_history: list[float] = []

    def on_market_state(self, state: MarketStateVector) -> None:
        pass

    def _raw_score(self, state: MarketStateVector) -> float:
        """Combine range expansion, volatility spike, price position in range."""
        # Range expansion: current range / avg range, >1 = expanding
        range_exp = getattr(state, "range_expansion", 1.0)
        expansion_score = (range_exp - 1.0) * 2.0  # 0 when normal, + when expanding

        # Volatility spike: vol_state already normalized
        vol_spike = state.volatility_state - 0.5  # Center at 0.5

        # Price vs recent high/low: 1 = at high (breakout up), 0 = at low (breakout down)
        price_pos = getattr(state, "price_vs_recent_high_low", 0.5)
        price_score = 2.0 * (price_pos - 0.5)  # -1 to 1

        # Breakout setup: expansion + vol -> then direction from price position
        setup = max(0.0, expansion_score) * (1.0 + vol_spike)
        raw = setup * price_score
        return max(-1.0, min(1.0, raw))

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal:
        raw = self._raw_score(state)
        self._score_history.append(raw)
        if len(self._score_history) > SCORE_NORMALIZE_WINDOW:
            self._score_history.pop(0)

        if len(self._score_history) >= 2:
            mean = sum(self._score_history) / len(self._score_history)
            std = max(
                math.sqrt(
                    sum((x - mean) ** 2 for x in self._score_history)
                    / (len(self._score_history) - 1)
                ),
                _EPS,
            )
            normalized = (raw - mean) / std if std >= _EPS else 0.0
            score = max(-1.0, min(1.0, math.tanh(normalized)))
        else:
            score = raw

        confidence = max(0.2, min(1.0, 0.5 * state.compression_state + 0.5 * state.volatility_state))

        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self.__class__.__name__,
            score=score,
            confidence=confidence,
        )
