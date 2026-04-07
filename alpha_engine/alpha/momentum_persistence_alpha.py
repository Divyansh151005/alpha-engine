"""Short-only momentum persistence: three consecutive down-bar returns -> strong short."""

from __future__ import annotations

import math
from collections import deque

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.state_engine.state_models import MarketStateVector


class MomentumPersistenceAlphaStrategy(BaseAlphaStrategy):
    """
    r1 = latest bar return, r2 = one bar ago, r3 = two bars ago (from ``latest_bar_return`` history).

    Emits a short signal only when all three are negative; never emits a long.
    """

    def __init__(self) -> None:
        self._bar_returns: deque[float] = deque(maxlen=512)

    def on_market_state(self, state: MarketStateVector) -> None:
        br = float(getattr(state, "latest_bar_return", float("nan")))
        if math.isfinite(br):
            self._bar_returns.append(br)

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        if len(self._bar_returns) < 3:
            return None
        r1 = self._bar_returns[-1]
        r2 = self._bar_returns[-2]
        r3 = self._bar_returns[-3]
        if not (math.isfinite(r1) and math.isfinite(r2) and math.isfinite(r3)):
            return None
        if r1 < 0.0 and r2 < 0.0 and r3 < 0.0:
            return AlphaSignal(
                timestamp=state.timestamp,
                strategy_name=self.__class__.__name__,
                score=-1.0,
                confidence=1.0,
            )
        return None
