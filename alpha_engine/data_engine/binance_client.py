"""Binance historical data client using ccxt."""

from __future__ import annotations

import time
from typing import Any

import ccxt

from alpha_engine.core.logger import get_logger
from alpha_engine.data.models import OHLCV

logger = get_logger(__name__)


class BinanceHistoricalClient:
    """Fetch historical OHLCV candles from Binance via ccxt."""

    def __init__(
        self,
        exchange: Any | None = None,
        max_retries: int = 3,
        retry_delay_sec: float = 1.0,
    ) -> None:
        self._exchange = exchange or ccxt.binance({"enableRateLimit": True})
        self._max_retries = max_retries
        self._retry_delay_sec = retry_delay_sec

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[OHLCV]:
        """Fetch latest candles for symbol/timeframe."""
        rows = self._run_with_retries(
            lambda: self._exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
        )
        return self._to_ohlcv(symbol=symbol, timeframe=timeframe, rows=rows)

    def fetch_since(
        self,
        symbol: str,
        timeframe: str,
        since_timestamp: int,
    ) -> list[OHLCV]:
        """Fetch candles from a given timestamp."""
        rows = self._run_with_retries(
            lambda: self._exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_timestamp,
            )
        )
        return self._to_ohlcv(symbol=symbol, timeframe=timeframe, rows=rows)

    def _run_with_retries(self, operation: Any) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "binance request failed attempt=%s/%s error=%s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay_sec)
        assert last_error is not None
        logger.exception("binance request exhausted retries")
        raise last_error

    @staticmethod
    def _to_ohlcv(symbol: str, timeframe: str, rows: list[list[float]]) -> list[OHLCV]:
        candles: list[OHLCV] = []
        for row in rows:
            candles.append(
                OHLCV(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        return candles
