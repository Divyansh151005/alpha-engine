"""Live Binance websocket stream for kline candles."""

from __future__ import annotations

import asyncio
import json

import websockets

from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.logger import get_logger
from alpha_engine.data.models import OHLCV

logger = get_logger(__name__)


class BinanceWebSocketStream:
    """Subscribe to Binance kline streams and publish candle events."""

    def __init__(self, event_bus: EventBus, base_url: str | None = None) -> None:
        self._event_bus = event_bus
        self._base_url = base_url or "wss://stream.binance.com:9443/ws"

    async def stream(
        self,
        symbol: str,
        timeframe: str,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Run a reconnecting live stream loop."""
        stream_name = self._build_stream_name(symbol=symbol, timeframe=timeframe)
        url = f"{self._base_url}/{stream_name}"

        while stop_event is None or not stop_event.is_set():
            try:
                async with websockets.connect(url) as socket:
                    async for raw_message in socket:
                        candle = self._parse_kline_message(
                            raw_message=raw_message,
                            symbol=symbol,
                            timeframe=timeframe,
                        )
                        if candle is None:
                            continue
                        self._event_bus.publish("market.candle", candle)
                        if stop_event is not None and stop_event.is_set():
                            break
            except Exception as exc:
                if stop_event is not None and stop_event.is_set():
                    break
                logger.exception(
                    "websocket stream error symbol=%s timeframe=%s error=%s",
                    symbol,
                    timeframe,
                    exc,
                )
                await asyncio.sleep(2.0)

    @staticmethod
    def _build_stream_name(symbol: str, timeframe: str) -> str:
        normalized_symbol = symbol.replace("/", "").lower()
        return f"{normalized_symbol}@kline_{timeframe}"

    @staticmethod
    def _parse_kline_message(
        raw_message: str,
        symbol: str,
        timeframe: str,
    ) -> OHLCV | None:
        payload = json.loads(raw_message)
        kline = payload.get("k")
        if not kline:
            return None

        return OHLCV(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=int(kline["t"]),
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
        )
