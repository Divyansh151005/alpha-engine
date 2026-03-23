"""Microstructure and short-horizon market-state features."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def range_compression(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    window: int,
) -> float:
    """Measure latest range relative to rolling average range."""
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
