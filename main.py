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
from alpha_engine.alpha import (
    AlphaEngine,
    AlphaSignal,
    BreakoutAlphaStrategy,
    MeanReversionAlphaStrategy,
    TrendAlphaStrategy,
)
from alpha_engine.volatility_engine import VolatilityTargetingEngine
from alpha_engine.risk_engine import RiskEngine
from alpha_engine.core.logger import get_logger
from alpha_engine.state_engine import MarketStateEngine, MarketStateVector
from alpha_engine.backtest import BacktestEngine

run_backtest = True
# Phase 8.6: Force signal injection debug (disabled for Phase 9.1 alpha visibility)
force_signal_debug = False
force_full_exposure = True
min_vol_scale = 0.5


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
    strategies = [
        TrendAlphaStrategy(),
        MeanReversionAlphaStrategy(),
        BreakoutAlphaStrategy(),
    ]
    AlphaEngine(event_bus=event_bus, strategies=strategies)
    CapitalAllocator(event_bus=event_bus)
    VolatilityTargetingEngine(event_bus=event_bus, target_vol=0.3)
    RiskEngine(event_bus=event_bus)

    def _on_features(feature_vector: FeatureVector) -> None:
        logger.info("Feature vector computed timestamp=%s", feature_vector.timestamp)

    event_bus.subscribe("market.features", _on_features)

    def _on_market_state(market_state_vector: MarketStateVector) -> None:
        logger.info("Market state computed timestamp=%s", market_state_vector.timestamp)

    event_bus.subscribe("market.state", _on_market_state)

    def _on_alpha_signal(signal: AlphaSignal) -> None:
        logger.info(
            "alpha.strategy name=%s signal=%.4f direction=%s",
            signal.strategy_name,
            signal.strength,
            signal.direction,
        )

    event_bus.subscribe("alpha.signal", _on_alpha_signal)

    def _on_allocation(allocations: list[PositionAllocation]) -> None:
        logger.info("Capital allocation computed count=%s", len(allocations))

    event_bus.subscribe("portfolio.allocation", _on_allocation)

    def _on_position(_positions: list[PositionAllocation]) -> None:
        logger.info("Volatility adjusted position computed")

    event_bus.subscribe("portfolio.position", _on_position)

    def _on_final(_positions: list[PositionAllocation]) -> None:
        logger.info("Final risk-adjusted position computed")

    event_bus.subscribe("portfolio.final", _on_final)

    storage = PostgresStorage()
    storage.init_schema()
    binance_client = BinanceHistoricalClient()
    data_service = DataService(
        event_bus=event_bus,
        storage=storage,
        binance_client=binance_client,
    )

    bootstrap_candles = data_service.bootstrap(symbol="BTC/USDT", timeframe="1m", limit=300)
    for candle in bootstrap_candles:
        event_bus.publish("market.candle", candle)

    logger.info("Historical bootstrap complete")

    if run_backtest and bootstrap_candles:
        backtest_engine = BacktestEngine(
            initial_equity=100.0,
            diagnostics=True,
            force_signal_debug=force_signal_debug,
            force_full_exposure=force_full_exposure,
            min_vol_scale=min_vol_scale,
            alpha_visibility=True,
        )
        result = backtest_engine.run(bootstrap_candles)
        logger.info("Backtest sanity: total_trades=%s avg_position=%.4f time_in_market=%.1f%%",
                    result.metrics.get("trade_count", 0),
                    result.metrics.get("avg_position_size", 0),
                    result.metrics.get("time_in_market", 0))
        print("Backtest metrics:")
        for key in ("total_return", "sharpe_ratio", "trade_count", "avg_position_size", "time_in_market"):
            if key in result.metrics:
                val = result.metrics[key]
                if key == "time_in_market":
                    print(f"  {key}: {val:.1f}%")
                elif key == "trade_count":
                    print(f"  {key}: {int(val)}")
                else:
                    print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()

