"""Tests for market data engine components."""

from __future__ import annotations

import asyncio

from alpha_engine.core.event_bus import EventBus
from alpha_engine.data.models import OHLCV
from alpha_engine.data_engine.binance_client import BinanceHistoricalClient
from alpha_engine.data_engine.data_service import DataService
from alpha_engine.data_engine.storage import PostgresStorage
from alpha_engine.data_engine.websocket_stream import BinanceWebSocketStream


class _FakeExchange:
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
        since: int | None = None,
    ) -> list[list[float]]:
        if since is not None:
            base = since
        else:
            base = 1700000000000
        size = limit or 2
        rows: list[list[float]] = []
        for i in range(size):
            rows.append(
                [
                    base + (i * 60000),
                    100.0 + i,
                    101.0 + i,
                    99.0 + i,
                    100.5 + i,
                    12.0 + i,
                ]
            )
        return rows


class _FakeCursor:
    def __init__(self, connection: "_FakeConnection") -> None:
        self._connection = connection
        self._results: list[tuple[object, ...]] = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(query.split()).lower()
        if normalized.startswith("insert into ohlcv"):
            assert params is not None
            key = (params[0], params[1], params[2])
            if key not in self._connection.rows:
                self._connection.rows[key] = tuple(params)
            return

        if normalized.startswith("select symbol, timeframe, timestamp"):
            assert params is not None
            symbol, timeframe, start_ts, end_ts = params
            filtered = [
                row
                for row in self._connection.rows.values()
                if row[0] == symbol
                and row[1] == timeframe
                and int(row[2]) >= int(start_ts)
                and int(row[2]) <= int(end_ts)
            ]
            self._results = sorted(filtered, key=lambda r: int(r[2]))
            return

        # CREATE statements from schema init are no-ops for this fake DB.
        if normalized.startswith("create table") or normalized.startswith("create index"):
            return

    def executemany(self, query: str, params_list: list[tuple[object, ...]]) -> None:
        for params in params_list:
            self.execute(query, params)

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._results


class _FakeConnection:
    def __init__(self) -> None:
        self.rows: dict[tuple[object, object, object], tuple[object, ...]] = {}

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)


def _sample_candle(timestamp: int = 1700000000000) -> OHLCV:
    return OHLCV(
        symbol="BTC/USDT",
        timeframe="1m",
        timestamp=timestamp,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=12.0,
    )


def test_historical_fetch_returns_candles() -> None:
    client = BinanceHistoricalClient(exchange=_FakeExchange())
    candles = client.fetch_ohlcv(symbol="BTC/USDT", timeframe="1m", limit=3)

    assert len(candles) == 3
    assert candles[0].symbol == "BTC/USDT"
    assert candles[0].timeframe == "1m"
    assert candles[0].timestamp == 1700000000000


def test_storage_insert_and_retrieval_works(monkeypatch) -> None:
    fake_connection = _FakeConnection()
    monkeypatch.setattr(
        "alpha_engine.data_engine.storage.psycopg2.connect",
        lambda *args, **kwargs: fake_connection,
    )

    storage = PostgresStorage()
    storage.init_schema()
    storage.insert_ohlcv(_sample_candle())

    candles = storage.get_range(
        symbol="BTC/USDT",
        timeframe="1m",
        start_ts=1699999990000,
        end_ts=1700000010000,
    )

    assert len(candles) == 1
    assert candles[0].close == 100.5


def test_duplicate_insert_does_not_fail(monkeypatch) -> None:
    fake_connection = _FakeConnection()
    monkeypatch.setattr(
        "alpha_engine.data_engine.storage.psycopg2.connect",
        lambda *args, **kwargs: fake_connection,
    )

    storage = PostgresStorage()
    candle = _sample_candle()
    storage.insert_ohlcv(candle)
    storage.insert_ohlcv(candle)

    candles = storage.get_range(
        symbol="BTC/USDT",
        timeframe="1m",
        start_ts=1699999990000,
        end_ts=1700000010000,
    )
    assert len(candles) == 1


def test_event_bus_publishes_candle_event() -> None:
    class _StorageStub:
        def __init__(self) -> None:
            self.saved: list[OHLCV] = []

        def insert_ohlcv(self, candle: OHLCV) -> None:
            self.saved.append(candle)

    class _ClientStub:
        def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> list[OHLCV]:
            return []

    bus = EventBus()
    storage = _StorageStub()
    service = DataService(event_bus=bus, storage=storage, binance_client=_ClientStub())
    published: list[OHLCV] = []
    bus.subscribe("market.candle", lambda candle: published.append(candle))

    candle = _sample_candle()
    service.handle_candle(candle)

    assert storage.saved == [candle]
    assert published == [candle]


def test_websocket_stream_publishes_parsed_candle(monkeypatch) -> None:
    class _FakeSocket:
        def __init__(self, message: str, stop_event: asyncio.Event) -> None:
            self._message = message
            self._stop_event = stop_event
            self._sent = False

        def __aiter__(self) -> "_FakeSocket":
            return self

        async def __anext__(self) -> str:
            if self._sent:
                raise StopAsyncIteration
            self._sent = True
            self._stop_event.set()
            return self._message

    class _FakeConnect:
        def __init__(self, socket: _FakeSocket) -> None:
            self._socket = socket

        async def __aenter__(self) -> _FakeSocket:
            return self._socket

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    stop_event = asyncio.Event()
    message = (
        '{"e":"kline","k":{"t":1700000000000,"o":"100.0","h":"101.0",'
        '"l":"99.0","c":"100.5","v":"12.0","i":"1m"}}'
    )
    fake_socket = _FakeSocket(message=message, stop_event=stop_event)
    monkeypatch.setattr(
        "alpha_engine.data_engine.websocket_stream.websockets.connect",
        lambda _: _FakeConnect(fake_socket),
    )

    bus = EventBus()
    stream = BinanceWebSocketStream(event_bus=bus)
    received: list[OHLCV] = []
    bus.subscribe("market.candle", lambda candle: received.append(candle))

    asyncio.run(stream.stream(symbol="BTC/USDT", timeframe="1m", stop_event=stop_event))

    assert len(received) == 1
    assert received[0].symbol == "BTC/USDT"
    assert received[0].timestamp == 1700000000000
