"""PostgreSQL storage implementation for OHLCV market data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extensions import connection as PgConnection

from alpha_engine.core.logger import get_logger
from alpha_engine.data.models import OHLCV

logger = get_logger(__name__)


class PostgresStorage:
    """Persist and query OHLCV candles from PostgreSQL."""

    def __init__(self, dsn: str | None = None, schema_path: str | None = None) -> None:
        self._dsn = dsn
        self._schema_path = schema_path or str(Path(__file__).with_name("schema.sql"))
        self._conn: PgConnection | None = None

    def connect(self) -> None:
        """Create a DB connection using DSN or environment variables."""
        if self._conn is not None:
            return

        try:
            if self._dsn:
                self._conn = psycopg2.connect(self._dsn)
            else:
                host = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST", "localhost")
                port = os.getenv("POSTGRES_PORT") or os.getenv("DB_PORT", "5432")
                user = os.getenv("POSTGRES_USER") or os.getenv("DB_USER", "postgres")
                password = os.getenv("POSTGRES_PASSWORD") or os.getenv("DB_PASSWORD", "")
                dbname = os.getenv("POSTGRES_DB") or os.getenv("DB_NAME", "postgres")
                self._conn = psycopg2.connect(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    dbname=dbname,
                )
        except Exception:
            logger.exception("failed to connect to postgres")
            raise

    def init_schema(self) -> None:
        """Create required schema objects if they do not exist."""
        self.connect()
        assert self._conn is not None

        schema_sql = Path(self._schema_path).read_text(encoding="utf-8")
        try:
            with self._conn:
                with self._conn.cursor() as cursor:
                    cursor.execute(schema_sql)
        except Exception:
            logger.exception("failed to initialize postgres schema")
            raise

    def insert_ohlcv(self, candle: OHLCV) -> None:
        """Insert one candle, ignoring duplicates."""
        self.connect()
        assert self._conn is not None

        query = """
            INSERT INTO ohlcv(symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
        """
        values = (
            candle.symbol,
            candle.timeframe,
            candle.timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )

        try:
            with self._conn:
                with self._conn.cursor() as cursor:
                    cursor.execute(query, values)
        except Exception:
            logger.exception(
                "failed to insert candle symbol=%s timeframe=%s timestamp=%s",
                candle.symbol,
                candle.timeframe,
                candle.timestamp,
            )
            raise

    def insert_batch(self, candles: list[OHLCV]) -> None:
        """Insert a batch of candles in one transaction."""
        if not candles:
            return

        self.connect()
        assert self._conn is not None

        query = """
            INSERT INTO ohlcv(symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
        """
        values: list[tuple[Any, ...]] = [
            (
                candle.symbol,
                candle.timeframe,
                candle.timestamp,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            )
            for candle in candles
        ]

        try:
            with self._conn:
                with self._conn.cursor() as cursor:
                    cursor.executemany(query, values)
        except Exception:
            logger.exception("failed to insert batch size=%s", len(candles))
            raise

    def get_range(
        self,
        symbol: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> list[OHLCV]:
        """Fetch candles in [start_ts, end_ts] for symbol/timeframe."""
        self.connect()
        assert self._conn is not None

        query = """
            SELECT symbol, timeframe, timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = %s
              AND timeframe = %s
              AND timestamp >= %s
              AND timestamp <= %s
            ORDER BY timestamp ASC
        """
        try:
            with self._conn.cursor() as cursor:
                cursor.execute(query, (symbol, timeframe, start_ts, end_ts))
                rows = cursor.fetchall()
        except Exception:
            logger.exception(
                "failed to fetch candle range symbol=%s timeframe=%s start=%s end=%s",
                symbol,
                timeframe,
                start_ts,
                end_ts,
            )
            raise

        return [
            OHLCV(
                symbol=row[0],
                timeframe=row[1],
                timestamp=int(row[2]),
                open=float(row[3]),
                high=float(row[4]),
                low=float(row[5]),
                close=float(row[6]),
                volume=float(row[7]),
            )
            for row in rows
        ]
