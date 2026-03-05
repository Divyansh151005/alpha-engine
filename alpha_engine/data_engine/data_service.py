"""Service orchestrating historical bootstrap and live data handling."""

from __future__ import annotations

import asyncio

from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.logger import get_logger
from alpha_engine.data.models import OHLCV
from alpha_engine.data_engine.binance_client import BinanceHistoricalClient
from alpha_engine.data_engine.storage import PostgresStorage
from alpha_engine.data_engine.websocket_stream import BinanceWebSocketStream

logger = get_logger(__name__)


class DataService:
    """Coordinate market data ingestion from Binance into storage."""

    def __init__(
        self,
        event_bus: EventBus,
        storage: PostgresStorage,
        binance_client: BinanceHistoricalClient,
    ) -> None:
        self._event_bus = event_bus
        self._storage = storage
        self._binance_client = binance_client
        self._stream = BinanceWebSocketStream(event_bus=self._event_bus)
        self._last_timestamps: dict[tuple[str, str], int] = {}

    def bootstrap(self, symbol: str, timeframe: str, limit: int) -> list[OHLCV]:
        """Fetch recent historical candles and persist them."""
        candles = self._binance_client.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )
        self._storage.insert_batch(candles)

        if candles:
            key = (symbol, timeframe)
            self._last_timestamps[key] = candles[-1].timestamp

        return candles

    def start_live_stream(self, symbol: str, timeframe: str) -> None:
        """Start websocket stream and process incoming candle events."""
        self._event_bus.subscribe("market.candle", self._stream_ingest_handler)
        asyncio.run(self._stream.stream(symbol=symbol, timeframe=timeframe))

    def handle_candle(self, candle: OHLCV) -> None:
        """Validate, persist, and publish a processed candle event."""
        self._validate_timestamp(candle)
        self._storage.insert_ohlcv(candle)
        self._event_bus.publish("market.candle", candle)

    def _stream_ingest_handler(self, candle: OHLCV) -> None:
        """Ingest stream candles without recursive event emission."""
        self._validate_timestamp(candle)
        self._storage.insert_ohlcv(candle)
        self._event_bus.publish("market.candle.stored", candle)

    def _validate_timestamp(self, candle: OHLCV) -> None:
        key = (candle.symbol, candle.timeframe)
        last_ts = self._last_timestamps.get(key)
        if last_ts is not None and candle.timestamp <= last_ts:
            logger.warning(
                "out-of-order candle rejected symbol=%s timeframe=%s timestamp=%s last=%s",
                candle.symbol,
                candle.timeframe,
                candle.timestamp,
                last_ts,
            )
            raise ValueError(
                f"candle timestamp {candle.timestamp} is not newer than {last_ts}"
            )

        self._last_timestamps[key] = candle.timestamp
