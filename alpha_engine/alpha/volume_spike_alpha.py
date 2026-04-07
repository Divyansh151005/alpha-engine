"""Volume spike alpha: discrete long/flat/short from volume vs rolling mean."""

from __future__ import annotations

import math

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.state_engine.state_models import MarketStateVector


class VolumeSpikeAlphaStrategy(BaseAlphaStrategy):
    """
    volume_ratio = volume / SMA(volume, 20) — supplied as ``MarketStateVector.volume_spike``
    from the feature pipeline (same definition).
    """

    def on_market_state(self, state: MarketStateVector) -> None:
        pass

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        volume_ratio = float(getattr(state, "volume_spike", 1.0))
        if not math.isfinite(volume_ratio) or volume_ratio <= 0.0:
            return None

        if volume_ratio > 2.0:
            score = 1.0
        elif volume_ratio < 0.5:
            score = -1.0
        else:
            score = 0.0

        confidence = 0.3 + 0.7 * abs(score)
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self.__class__.__name__,
            score=score,
            confidence=confidence,
        )
