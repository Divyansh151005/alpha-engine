"""Alpha Engine entry point."""

from __future__ import annotations

from dotenv import load_dotenv

from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.config import get_settings
from alpha_engine.data_engine.binance_client import BinanceHistoricalClient
from alpha_engine.data_engine.data_service import DataService
from alpha_engine.data_engine.storage import PostgresStorage
from alpha_engine.features import FeatureEngine, FeatureVector
from alpha_engine.allocation_engine import CapitalAllocator, PositionAllocation
from alpha_engine.alpha import AlphaEngine, AlphaSignal, TrendAlphaStrategy
from alpha_engine.core.logger import get_logger
from alpha_engine.state_engine import MarketStateEngine, MarketStateVector


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
    MarketStateEngine(event_bus=event_bus)
    AlphaEngine(event_bus=event_bus, strategies=[TrendAlphaStrategy()])
    CapitalAllocator(event_bus=event_bus)

    def _on_features(feature_vector: FeatureVector) -> None:
        logger.info("Feature vector computed timestamp=%s", feature_vector.timestamp)

    event_bus.subscribe("market.features", _on_features)

    def _on_market_state(market_state_vector: MarketStateVector) -> None:
        logger.info("Market state computed timestamp=%s", market_state_vector.timestamp)

    event_bus.subscribe("market.state", _on_market_state)

    def _on_alpha_signal(signal: AlphaSignal) -> None:
        logger.info(
            "Alpha signal generated strategy=%s direction=%s strength=%.4f confidence=%.4f timestamp=%s",
            signal.strategy_name,
            signal.direction,
            signal.strength,
            signal.confidence,
            signal.timestamp,
        )

    event_bus.subscribe("alpha.signal", _on_alpha_signal)

    def _on_allocation(allocations: list[PositionAllocation]) -> None:
        logger.info("Capital allocation computed count=%s", len(allocations))

    event_bus.subscribe("portfolio.allocation", _on_allocation)

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

