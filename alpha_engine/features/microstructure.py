"""Microstructure and short-horizon market-state features."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def range_compression(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    window: int,
) -> float:
    """Measure latest range relative to rolling average range. <1 = compressed, >1 = expanded."""
    if window <= 0:
        raise ValueError("window must be positive")

    highs = np.asarray(high, dtype=np.float64)
    lows = np.asarray(low, dtype=np.float64)
    if highs.size != lows.size:
        raise ValueError("high and low must have equal lengths")
    if highs.size < window:
        return float("nan")

    ranges = highs - lows
    recent = ranges[-window:]
    avg_range = float(np.mean(recent))
    if abs(avg_range) < _EPS:
        return 0.0
    return float(recent[-1] / avg_range)


def range_expansion(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    window: int,
) -> float:
    """Range expansion ratio: current range / rolling avg. >1 = expanding (breakout setup)."""
    return range_compression(high, low, window)


def price_vs_recent_high_low(
    close: np.ndarray | list[float],
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    window: int,
) -> float:
    """Position within recent range: 1 = at high, 0 = at low, 0.5 = middle."""
    if window <= 0:
        raise ValueError("window must be positive")
    c = np.asarray(close, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l_arr = np.asarray(low, dtype=np.float64)
    if c.size < window or h.size < window or l_arr.size < window:
        return float("nan")
    recent_high = float(np.max(h[-window:]))
    recent_low = float(np.min(l_arr[-window:]))
    price = float(c[-1])
    rng = recent_high - recent_low
    if rng < _EPS:
        return 0.5
    return max(0.0, min(1.0, (price - recent_low) / rng))


def volume_spike(volume: np.ndarray | list[float], window: int) -> float:
    """Measure latest volume relative to rolling average volume."""
    if window <= 0:
        raise ValueError("window must be positive")

    values = np.asarray(volume, dtype=np.float64)
    if values.size < window:
        return float("nan")

    recent = values[-window:]
    baseline = float(np.mean(recent))
    if abs(baseline) < _EPS:
        return 0.0
    return float(recent[-1] / baseline)


def lower_wick_ratio(open_: float, high: float, low: float, close: float) -> float:
    """Lower wick size as fraction of candle range. Long lower wick -> high value."""
    candle_range = high - low
    if candle_range < _EPS:
        return 0.0
    body_bottom = min(open_, close)
    wick_size = body_bottom - low
    return max(0.0, min(1.0, float(wick_size / candle_range)))


def upper_wick_ratio(open_: float, high: float, low: float, close: float) -> float:
    """Upper wick size as fraction of candle range. Long upper wick -> high value."""
    candle_range = high - low
    if candle_range < _EPS:
        return 0.0
    body_top = max(open_, close)
    wick_size = high - body_top
    return max(0.0, min(1.0, float(wick_size / candle_range)))


def return_zscore(returns: np.ndarray | list[float], window: int) -> float:
    """Z-score of latest return against rolling return distribution."""
    if window <= 1:
        raise ValueError("window must be greater than one")

    values = np.asarray(returns, dtype=np.float64)
    if values.size < window:
        return float("nan")

    recent = values[-window:]
    mean = float(np.mean(recent))
    std = float(np.std(recent))
    if std < _EPS:
        return 0.0
    return float((recent[-1] - mean) / std)


def mean_reversion_strength(
    close: np.ndarray | list[float],
    window: int = 20,
) -> float:
    """Distance from rolling mean as fraction of rolling std. High = stretched, expect reversion."""
    if window <= 1:
        raise ValueError("window must be > 1")
    values = np.asarray(close, dtype=np.float64)
    if values.size < window:
        return float("nan")
    recent = values[-window:]
    mean = float(np.mean(recent))
    std = float(np.std(recent))
    if std < _EPS:
        return 0.0
    deviation = (recent[-1] - mean) / std
    return float(deviation)


def bollinger_band_distance(
    close: np.ndarray | list[float],
    window: int = 20,
    num_std: float = 2.0,
) -> float:
    """Signed distance from price to middle band, in units of band width. >0 = above, <0 = below."""
    if window <= 1:
        raise ValueError("window must be > 1")
    values = np.asarray(close, dtype=np.float64)
    if values.size < window:
        return float("nan")
    recent = values[-window:]
    mean = float(np.mean(recent))
    std = float(np.std(recent))
    if std < _EPS:
        return 0.0
    price = recent[-1]
    upper = mean + num_std * std
    lower = mean - num_std * std
    band_width = upper - lower
    if band_width < _EPS:
        return 0.0
    # Distance: +1 = at upper band, -1 = at lower band
    return float((price - mean) / (num_std * std))
