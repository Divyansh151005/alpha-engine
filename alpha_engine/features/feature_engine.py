"""Event-driven feature computation layer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from alpha_engine.core.event_bus import EventBus
from alpha_engine.data.models import OHLCV
from alpha_engine.features.microstructure import (
    lower_wick_ratio,
    range_compression,
    return_zscore,
    upper_wick_ratio,
    volume_spike,
)
from alpha_engine.features.trend import ema, momentum, trend_strength
from alpha_engine.features.volatility import atr, calculate_returns, realized_volatility


@dataclass(frozen=True)
class FeatureVector:
    """Computed market-state features at a given timestamp."""

    timestamp: int
    realized_vol: float
    atr: float
    trend_strength: float
    momentum: float
    range_compression: float
    volume_spike: float
    return_zscore: float
    lower_wick_ratio: float
    upper_wick_ratio: float


class FeatureEngine:
    """Compute and publish feature vectors from incoming candle events."""

    VOLATILITY_WINDOW = 20
    ATR_WINDOW = 14
    EMA_SHORT_PERIOD = 12
    EMA_LONG_PERIOD = 26
    MOMENTUM_WINDOW = 10
    RANGE_WINDOW = 20
    VOLUME_WINDOW = 20
    RETURN_ZSCORE_WINDOW = 20

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._required_candles = max(
            self.EMA_LONG_PERIOD,
            self.MOMENTUM_WINDOW + 1,
            self.VOLATILITY_WINDOW + 1,
            self.RETURN_ZSCORE_WINDOW + 1,
            self.ATR_WINDOW,
            self.RANGE_WINDOW,
            self.VOLUME_WINDOW,
        )
        self._open: deque[float] = deque(maxlen=self._required_candles)
        self._close: deque[float] = deque(maxlen=self._required_candles)
        self._high: deque[float] = deque(maxlen=self._required_candles)
        self._low: deque[float] = deque(maxlen=self._required_candles)
        self._volume: deque[float] = deque(maxlen=self._required_candles)

        self._event_bus.subscribe("market.candle", self._on_candle)

    def _on_candle(self, candle: OHLCV) -> None:
        self._open.append(candle.open)
        self._close.append(candle.close)
        self._high.append(candle.high)
        self._low.append(candle.low)
        self._volume.append(candle.volume)

        if len(self._close) < self._required_candles:
            return

        close_arr = np.asarray(self._close, dtype=np.float64)
        high_arr = np.asarray(self._high, dtype=np.float64)
        low_arr = np.asarray(self._low, dtype=np.float64)
        volume_arr = np.asarray(self._volume, dtype=np.float64)
        returns = calculate_returns(close_arr)

        o, h, l, c = self._open[-1], self._high[-1], self._low[-1], self._close[-1]
        low_wick = lower_wick_ratio(float(o), float(h), float(l), float(c))
        up_wick = upper_wick_ratio(float(o), float(h), float(l), float(c))

        feature_vector = FeatureVector(
            timestamp=candle.timestamp,
            realized_vol=realized_volatility(returns, self.VOLATILITY_WINDOW),
            atr=atr(high_arr, low_arr, close_arr, self.ATR_WINDOW),
            trend_strength=trend_strength(
                ema(close_arr, self.EMA_SHORT_PERIOD),
                ema(close_arr, self.EMA_LONG_PERIOD),
            ),
            momentum=momentum(close_arr, self.MOMENTUM_WINDOW),
            range_compression=range_compression(high_arr, low_arr, self.RANGE_WINDOW),
            volume_spike=volume_spike(volume_arr, self.VOLUME_WINDOW),
            return_zscore=return_zscore(returns, self.RETURN_ZSCORE_WINDOW),
            lower_wick_ratio=low_wick,
            upper_wick_ratio=up_wick,
        )

        if all(np.isfinite(value) for value in feature_vector.__dict__.values()):
            self._event_bus.publish("market.features", feature_vector)
