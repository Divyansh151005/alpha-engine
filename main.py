"""Alpha Engine entry point."""

from __future__ import annotations

from dotenv import load_dotenv

from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.config import get_settings
from alpha_engine.data_engine.binance_client import BinanceHistoricalClient
from alpha_engine.data_engine.data_service import DataService
from alpha_engine.data_engine.storage import PostgresStorage
from alpha_engine.features import FeatureEngine, FeatureVector
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
    FeatureEngine(event_bus=event_bus)

    def _on_features(feature_vector: FeatureVector) -> None:
        logger.info("Feature vector computed timestamp=%s", feature_vector.timestamp)

    event_bus.subscribe("market.features", _on_features)

    storage = PostgresStorage()
    storage.init_schema()
    binance_client = BinanceHistoricalClient()
    data_service = DataService(
        event_bus=event_bus,
        storage=storage,
        binance_client=binance_client,
    )

    bootstrap_candles = data_service.bootstrap(symbol="BTC/USDT", timeframe="1m", limit=100)
    for candle in bootstrap_candles:
        event_bus.publish("market.candle", candle)

    logger.info("Historical bootstrap complete")


if __name__ == "__main__":
    main()

