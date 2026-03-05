"""Core market data models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OHLCV:
    """Single candlestick observation."""

    symbol: str
    timeframe: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

