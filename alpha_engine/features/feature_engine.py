"""Event-driven feature computation layer."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from alpha_engine.core.event_bus import EventBus
from alpha_engine.data.models import OHLCV
from alpha_engine.features.microstructure import (
    bollinger_band_distance,
    lower_wick_ratio,
    mean_reversion_strength,
    price_vs_recent_high_low,
    range_compression,
    range_expansion,
    return_zscore,
    upper_wick_ratio,
    volume_spike,
)
from alpha_engine.features.trend import (
    ema,
    ema_crossover_strength,
    momentum,
    momentum_acceleration,
    trend_strength,
)
from alpha_engine.features.volatility import (
    atr,
    calculate_returns,
    realized_volatility,
    rolling_return,
    volatility_breakout_signal,
)

_ROLLING_RETURN_WINDOWS = (5, 10, 20)
_VOL_BREAKOUT_BASELINE = 20
_MOM_ACCEL_SHORT = 5
_MOM_ACCEL_LONG = 10
_EMA_CROSS_SHORT = 12
_EMA_CROSS_LONG = 26
_BOLLINGER_WINDOW = 20
_MR_STRENGTH_WINDOW = 20
_PRICE_VS_RANGE_WINDOW = 20


def compute_volatility(close_prices: np.ndarray | list[float], window: int = 20) -> float:
    returns = np.diff(np.asarray(close_prices, dtype=np.float64))
    prices = np.asarray(close_prices, dtype=np.float64)[:-1]
    if prices.size == 0:
        return 0.0
    denom = np.where(np.abs(prices) < 1e-12, np.nan, prices)
    rets = returns / denom
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.std(rets[-window:])) if rets.size >= window else 0.0


@dataclass(frozen=True)
class FeatureVector:
    """Computed market-state features at a given timestamp."""

    timestamp: int
    symbol: str | None
    realized_vol: float
    volatility: float
    atr: float
    trend_strength: float
    momentum: float
    range_compression: float
    volume_spike: float
    return_zscore: float
    lower_wick_ratio: float
    upper_wick_ratio: float
    # Phase 11 additions
    rolling_return_5: float
    rolling_return_10: float
    rolling_return_20: float
    regime_trend_strength: float
    latest_bar_return: float
    volatility_breakout: float
    momentum_acceleration: float
    mean_reversion_strength: float
    bollinger_band_distance: float
    range_expansion: float
    price_vs_recent_high_low: float
    ema_crossover_strength: float
    funding_rate: float
    oi_change: float


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
            max(_ROLLING_RETURN_WINDOWS) + 1,
            _MOM_ACCEL_LONG + 1,
            _VOL_BREAKOUT_BASELINE,
            _BOLLINGER_WINDOW,
            _MR_STRENGTH_WINDOW,
            _PRICE_VS_RANGE_WINDOW,
        )
        self._open: deque[float] = deque(maxlen=self._required_candles)
        self._close: deque[float] = deque(maxlen=self._required_candles)
        self._high: deque[float] = deque(maxlen=self._required_candles)
        self._low: deque[float] = deque(maxlen=self._required_candles)
        self._volume: deque[float] = deque(maxlen=self._required_candles)
        self._rolling_realized_vol: deque[float] = deque(maxlen=_VOL_BREAKOUT_BASELINE + 5)
        self._last_open_interest: float | None = None
        self._last_candle_symbol: str | None = None

        self._event_bus.subscribe("market.candle", self._on_candle)

    def _on_candle(self, candle: OHLCV) -> None:
        sym = candle.symbol
        if self._last_candle_symbol is not None and sym != self._last_candle_symbol:
            self._last_open_interest = None
        self._last_candle_symbol = sym

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
        latest_bar_return = float(returns[-1]) if returns.size > 0 else 0.0
        rr20_raw = rolling_return(close_arr, 20)
        regime_trend_strength = (
            float(abs(rr20_raw)) if np.isfinite(rr20_raw) else 0.0
        )
        rv = realized_volatility(returns, self.VOLATILITY_WINDOW)
        vol = compute_volatility(close_arr, self.VOLATILITY_WINDOW)
        self._rolling_realized_vol.append(rv)
        vol_breakout_val = (
            volatility_breakout_signal(np.asarray(self._rolling_realized_vol), _VOL_BREAKOUT_BASELINE)
            if len(self._rolling_realized_vol) >= _VOL_BREAKOUT_BASELINE
            else 1.0  # Neutral when insufficient history
        )

        o, h, l, c = self._open[-1], self._high[-1], self._low[-1], self._close[-1]
        low_wick = lower_wick_ratio(float(o), float(h), float(l), float(c))
        up_wick = upper_wick_ratio(float(o), float(h), float(l), float(c))

        _fr = getattr(candle, "funding_rate", None)
        funding_f = (
            float(_fr)
            if _fr is not None and math.isfinite(float(_fr))
            else float("nan")
        )

        _oi_raw = getattr(candle, "open_interest", None)
        oi_change = float("nan")
        if _oi_raw is not None and math.isfinite(float(_oi_raw)):
            cur_oi = float(_oi_raw)
            if (
                self._last_open_interest is not None
                and math.isfinite(self._last_open_interest)
                and self._last_open_interest > 0.0
            ):
                oi_change = (cur_oi - self._last_open_interest) / self._last_open_interest
            self._last_open_interest = cur_oi

        feature_vector = FeatureVector(
            timestamp=candle.timestamp,
            symbol=candle.symbol,
            realized_vol=rv,
            volatility=vol,
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
            rolling_return_5=rolling_return(close_arr, 5),
            rolling_return_10=rolling_return(close_arr, 10),
            rolling_return_20=rolling_return(close_arr, 20),
            regime_trend_strength=regime_trend_strength,
            latest_bar_return=latest_bar_return,
            volatility_breakout=vol_breakout_val,
            momentum_acceleration=momentum_acceleration(close_arr, _MOM_ACCEL_SHORT, _MOM_ACCEL_LONG),
            mean_reversion_strength=mean_reversion_strength(close_arr, _MR_STRENGTH_WINDOW),
            bollinger_band_distance=bollinger_band_distance(close_arr, _BOLLINGER_WINDOW),
            range_expansion=range_expansion(high_arr, low_arr, self.RANGE_WINDOW),
            price_vs_recent_high_low=price_vs_recent_high_low(close_arr, high_arr, low_arr, _PRICE_VS_RANGE_WINDOW),
            ema_crossover_strength=ema_crossover_strength(close_arr, _EMA_CROSS_SHORT, _EMA_CROSS_LONG),
            funding_rate=funding_f,
            oi_change=oi_change,
        )

        core = {
            k: v
            for k, v in feature_vector.__dict__.items()
            if k not in ("funding_rate", "oi_change", "symbol")
        }
        if all(np.isfinite(value) for value in core.values()):
            self._event_bus.publish("market.features", feature_vector)
