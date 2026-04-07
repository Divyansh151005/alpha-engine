"""Perpetual funding contrarian tilt: very negative funding → long; very positive → short.

Matches ``signal_funding_contrarian_z`` / ``funding_contrarian_score_tail`` in
``alpha_engine.data.funding_rates``. Prefer ``MarketStateVector.funding_rate`` from
OHLCV-aligned backtests; alternatively ``note_funding`` / ``note_funding_batch``.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.data.funding_rates import funding_contrarian_score_tail
from alpha_engine.state_engine.state_models import MarketStateVector


class FundingContrarianAlphaStrategy(BaseAlphaStrategy):
    """Rolling z of funding (negated); score clipped to [-1, 1] via tanh scaling."""

    def __init__(self, z_window: int = 168) -> None:
        if z_window < 10:
            raise ValueError("z_window must be >= 10")
        self._z_window = z_window
        self._buf: deque[float] = deque(maxlen=z_window + 1)

    def note_funding(self, rate: float) -> None:
        """Inject funding from an external feed when state does not carry it."""
        if math.isfinite(rate):
            self._buf.append(float(rate))

    def note_funding_batch(self, rates: list[float]) -> None:
        for r in rates:
            self.note_funding(r)

    def on_market_state(self, state: MarketStateVector) -> None:
        fr = float(getattr(state, "funding_rate", float("nan")))
        if math.isfinite(fr):
            self._buf.append(fr)

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        if len(self._buf) < self._z_window:
            return None
        tail = np.asarray(self._buf, dtype=np.float64)[-self._z_window :]
        raw = funding_contrarian_score_tail(tail)
        if not math.isfinite(raw):
            return None
        score = math.tanh(raw / 1.35)
        if abs(score) < 1e-6:
            return None
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self.__class__.__name__,
            score=score,
            confidence=min(1.0, abs(raw) / 3.0),
        )
