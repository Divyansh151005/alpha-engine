"""Trend and momentum indicators."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def ema(series: np.ndarray | list[float], period: int) -> float:
    """Compute the latest exponential moving average value."""
    if period <= 0:
        raise ValueError("period must be positive")

    values = np.asarray(series, dtype=np.float64)
    if values.size == 0:
        return float("nan")

    alpha = 2.0 / (period + 1.0)
    ema_value = values[0]
    for value in values[1:]:
        ema_value = (alpha * value) + ((1.0 - alpha) * ema_value)
    return float(ema_value)


def trend_strength(short_ema: float, long_ema: float) -> float:
    """Normalized EMA spread measuring directional trend strength."""
    if abs(long_ema) < _EPS:
        return 0.0
    return float((short_ema - long_ema) / long_ema)


def momentum(close_prices: np.ndarray | list[float], window: int) -> float:
    """Simple price momentum over a lookback window."""
    if window <= 0:
        raise ValueError("window must be positive")

    values = np.asarray(close_prices, dtype=np.float64)
    if values.size <= window:
        return float("nan")

    base = values[-(window + 1)]
    if abs(base) < _EPS:
        return 0.0
    return float((values[-1] / base) - 1.0)


def momentum_acceleration(
    close_prices: np.ndarray | list[float],
    short_window: int = 5,
    long_window: int = 10,
) -> float:
    """Change in momentum: short-period mom minus long-period mom (acceleration)."""
    if short_window <= 0 or long_window <= 0 or short_window >= long_window:
        raise ValueError("short_window < long_window required")
    values = np.asarray(close_prices, dtype=np.float64)
    if values.size <= long_window:
        return float("nan")
    mom_short = momentum(values, short_window)
    mom_long = momentum(values, long_window)
    if np.isnan(mom_short) or np.isnan(mom_long):
        return float("nan")
    return float(mom_short - mom_long)


def ema_crossover_strength(
    close_prices: np.ndarray | list[float],
    short_period: int = 12,
    long_period: int = 26,
) -> float:
    """Normalized spread (short EMA - long EMA) / long EMA. Positive = bullish crossover."""
    if short_period <= 0 or long_period <= 0:
        raise ValueError("periods must be positive")
    values = np.asarray(close_prices, dtype=np.float64)
    if values.size < long_period:
        return float("nan")
    short_ema_val = ema(values, short_period)
    long_ema_val = ema(values, long_period)
    if abs(long_ema_val) < _EPS:
        return 0.0
    return float((short_ema_val - long_ema_val) / long_ema_val)
