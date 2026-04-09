"""Tests for volatility targeting engine."""

from __future__ import annotations

from alpha_engine.allocation_engine import PositionAllocation
from alpha_engine.core.event_bus import EventBus
from alpha_engine.features.feature_engine import FeatureVector
from alpha_engine.volatility_engine import VolatilityTargetingEngine


def _feature_vector(realized_vol: float, timestamp: int = 1000) -> FeatureVector:
    return FeatureVector(
        timestamp=timestamp,
        symbol=None,
        realized_vol=realized_vol,
        volatility=realized_vol,
        atr=0.0,
        trend_strength=0.0,
        momentum=0.0,
        range_compression=0.0,
        volume_spike=0.0,
        return_zscore=0.0,
        lower_wick_ratio=0.0,
        upper_wick_ratio=0.0,
        rolling_return_5=0.0,
        rolling_return_10=0.0,
        rolling_return_20=0.0,
        regime_trend_strength=0.0,
        latest_bar_return=0.0,
        volatility_breakout=1.0,
        momentum_acceleration=0.0,
        mean_reversion_strength=0.0,
        bollinger_band_distance=0.0,
        range_expansion=1.0,
        price_vs_recent_high_low=0.5,
        ema_crossover_strength=0.0,
        funding_rate=float("nan"),
        oi_change=float("nan"),
    )


def _allocation(capital_fraction: float = 0.5) -> list[PositionAllocation]:
    return [
        PositionAllocation(
            timestamp=1000,
            strategy_name="test",
            direction="long",
            capital_fraction=capital_fraction,
            risk_weight=0.5,
        )
    ]


def test_scaling_works_correctly() -> None:
    """Scaling adjusts allocation proportionally to target_vol / current_vol."""
    event_bus = EventBus()
    VolatilityTargetingEngine(event_bus=event_bus, target_vol=0.2)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.position", received.append)

    event_bus.publish("market.features", _feature_vector(realized_vol=0.1))
    event_bus.publish("portfolio.allocation", _allocation(capital_fraction=0.5))

    assert len(received) == 1
    adjusted = received[0][0]
    # scale = 0.2 / 0.1 = 2.0, so 0.5 * 2.0 = 1.0 (clipped)
    assert adjusted.capital_fraction == 1.0
    assert adjusted.strategy_name == "test"


def test_high_vol_reduces_allocation() -> None:
    """High volatility reduces position size."""
    event_bus = EventBus()
    VolatilityTargetingEngine(event_bus=event_bus, target_vol=0.2)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.position", received.append)

    event_bus.publish("market.features", _feature_vector(realized_vol=0.4))
    event_bus.publish("portfolio.allocation", _allocation(capital_fraction=0.8))

    assert len(received) == 1
    adjusted = received[0][0]
    # scale = 0.2 / 0.4 = 0.5, so 0.8 * 0.5 = 0.4
    assert adjusted.capital_fraction == 0.4
    assert adjusted.capital_fraction < 0.8


def test_low_vol_increases_allocation() -> None:
    """Low volatility increases position size."""
    event_bus = EventBus()
    VolatilityTargetingEngine(event_bus=event_bus, target_vol=0.3)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.position", received.append)

    event_bus.publish("market.features", _feature_vector(realized_vol=0.15))
    event_bus.publish("portfolio.allocation", _allocation(capital_fraction=0.3))

    assert len(received) == 1
    adjusted = received[0][0]
    # scale = 0.3 / 0.15 = 2.0, so 0.3 * 2.0 = 0.6
    assert adjusted.capital_fraction == 0.6
    assert adjusted.capital_fraction > 0.3


def test_no_division_by_zero() -> None:
    """Zero or missing volatility does not cause division by zero."""
    event_bus = EventBus()
    VolatilityTargetingEngine(event_bus=event_bus, target_vol=0.3)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.position", received.append)

    event_bus.publish("market.features", _feature_vector(realized_vol=0.0))
    event_bus.publish("portfolio.allocation", _allocation(capital_fraction=0.5))

    assert len(received) == 1
    adjusted = received[0][0]
    assert adjusted.capital_fraction == 0.5


def test_no_features_skips_scaling() -> None:
    """When no market.features received, allocation passes through unchanged."""
    event_bus = EventBus()
    VolatilityTargetingEngine(event_bus=event_bus, target_vol=0.3)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.position", received.append)

    event_bus.publish("portfolio.allocation", _allocation(capital_fraction=0.7))

    assert len(received) == 1
    adjusted = received[0][0]
    assert adjusted.capital_fraction == 0.7


def test_output_bounded() -> None:
    """Adjusted fractions are always in [0, 1]."""
    event_bus = EventBus()
    VolatilityTargetingEngine(event_bus=event_bus, target_vol=0.3)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.position", received.append)

    event_bus.publish("market.features", _feature_vector(realized_vol=0.001))
    event_bus.publish(
        "portfolio.allocation",
        [
            PositionAllocation(
                timestamp=1000,
                strategy_name="a",
                direction="long",
                capital_fraction=1.0,
                risk_weight=0.5,
            ),
        ],
    )

    assert len(received) == 1
    for a in received[0]:
        assert 0.0 <= a.capital_fraction <= 1.0


def test_very_small_vol_caps_scaling() -> None:
    """Very small volatility does not cause scaling explosion."""
    event_bus = EventBus()
    VolatilityTargetingEngine(event_bus=event_bus, target_vol=0.3)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.position", received.append)

    event_bus.publish("market.features", _feature_vector(realized_vol=1e-12))
    event_bus.publish("portfolio.allocation", _allocation(capital_fraction=0.5))

    assert len(received) == 1
    adjusted = received[0][0]
    assert 0.0 <= adjusted.capital_fraction <= 1.0
