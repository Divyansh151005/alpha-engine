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
    # Last known USDT-M funding rate at or before this bar's timestamp (no lookahead).
    funding_rate: float | None = None
    # Last known USDT-M open interest (contracts) at or before this bar's timestamp.
    open_interest: float | None = None

