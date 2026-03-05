"""Alpha Engine entry point."""

from __future__ import annotations

from dotenv import load_dotenv

from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.config import get_settings
from alpha_engine.data_engine.binance_client import BinanceHistoricalClient
from alpha_engine.data_engine.data_service import DataService
from alpha_engine.data_engine.storage import PostgresStorage
from alpha_engine.core.logger import get_logger


def main() -> None:
    load_dotenv()
    logger = get_logger(__name__)
    settings = get_settings()
    logger.info(
        "loaded settings exchange=%s symbol=%s",
        settings.exchange_name,
        settings.default_symbol,
    )

    event_bus = EventBus()
    storage = PostgresStorage()
    storage.init_schema()
    binance_client = BinanceHistoricalClient()
    data_service = DataService(
        event_bus=event_bus,
        storage=storage,
        binance_client=binance_client,
    )

    data_service.bootstrap(symbol="BTC/USDT", timeframe="1m", limit=100)
    logger.info("Historical bootstrap complete")


if __name__ == "__main__":
    main()

