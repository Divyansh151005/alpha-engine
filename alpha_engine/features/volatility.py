"""Volatility-oriented feature utilities."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def calculate_returns(prices: np.ndarray | list[float]) -> np.ndarray:
    """Compute simple returns from a price series."""
    values = np.asarray(prices, dtype=np.float64)
    if values.size < 2:
        return np.array([], dtype=np.float64)

    prev = values[:-1]
    denom = np.where(np.abs(prev) < _EPS, np.nan, prev)
    returns = np.diff(values) / denom
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)


def realized_volatility(returns: np.ndarray | list[float], window: int) -> float:
    """Compute windowed realized volatility from returns."""
    if window <= 0:
        raise ValueError("window must be positive")

    values = np.asarray(returns, dtype=np.float64)
    if values.size < window:
        return float("nan")

    recent = values[-window:]
    return float(np.sqrt(np.mean(np.square(recent))) * np.sqrt(window))


def atr(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    window: int,
) -> float:
    """Compute average true range (ATR) over the provided window."""
    if window <= 0:
        raise ValueError("window must be positive")

    highs = np.asarray(high, dtype=np.float64)
    lows = np.asarray(low, dtype=np.float64)
    closes = np.asarray(close, dtype=np.float64)

    if highs.size != lows.size or lows.size != closes.size:
        raise ValueError("high, low, and close must have equal lengths")
    if highs.size < window:
        return float("nan")

    true_ranges = np.empty_like(highs)
    true_ranges[0] = highs[0] - lows[0]
    if highs.size > 1:
        prev_close = closes[:-1]
        true_ranges[1:] = np.maximum.reduce(
            [
                highs[1:] - lows[1:],
                np.abs(highs[1:] - prev_close),
                np.abs(lows[1:] - prev_close),
            ]
        )

    return float(np.mean(true_ranges[-window:]))


def volatility_percentile(vol_series: np.ndarray | list[float]) -> float:
    """Percentile rank of the latest volatility point within the series."""
    values = np.asarray(vol_series, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")

    latest = finite[-1]
    rank = np.count_nonzero(finite <= latest)
    return float((rank / finite.size) * 100.0)
