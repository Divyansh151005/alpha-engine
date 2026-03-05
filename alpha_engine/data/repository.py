"""In-memory market data repository."""

from __future__ import annotations

from collections import defaultdict

from alpha_engine.data.models import OHLCV


class MarketDataRepository:
    """Repository API for storing and retrieving market data."""

    def __init__(self) -> None:
        self._store: dict[str, list[OHLCV]] = defaultdict(list)

    def save_ohlcv(self, symbol: str, record: OHLCV) -> None:
        """Persist one OHLCV record for a symbol."""
        self._store[symbol].append(record)

    def get_ohlcv(self, symbol: str) -> list[OHLCV]:
        """Return all stored OHLCV records for a symbol."""
        return list(self._store.get(symbol, []))

