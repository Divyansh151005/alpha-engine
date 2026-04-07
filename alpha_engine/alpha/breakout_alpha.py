"""Breakout alpha: volatility-normalized bar return spike (lightweight breakout detector)."""

from __future__ import annotations

import math
import statistics
from collections import deque

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.state_engine.state_models import MarketStateVector

_EPS = 1e-12
_RETURN_WINDOW = 20


class BreakoutAlphaStrategy(BaseAlphaStrategy):
    """Fire when |latest return| exceeds twice the recent return volatility (20-bar std)."""

    def __init__(self) -> None:
        self._recent_returns: deque[float] = deque(maxlen=_RETURN_WINDOW)

    def on_market_state(self, state: MarketStateVector) -> None:
        pass

    def _raw_score(self, state: MarketStateVector) -> float:
        recent_return = float(getattr(state, "latest_bar_return", 0.0))
        vol = (
            statistics.pstdev(self._recent_returns)
            if len(self._recent_returns) >= 2
            else 0.0
        )
        if math.isfinite(recent_return):
            self._recent_returns.append(recent_return)

        if not math.isfinite(vol) or vol <= _EPS:
            score = 0.0
        elif abs(recent_return) > 2.0 * vol:
            score = math.copysign(1.0, recent_return)
        else:
            score = 0.0
        return max(-1.0, min(1.0, score))

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal:
        score = self._raw_score(state)
        confidence = 0.3 + 0.7 * abs(score)
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self.__class__.__name__,
            score=score,
            confidence=confidence,
        )
