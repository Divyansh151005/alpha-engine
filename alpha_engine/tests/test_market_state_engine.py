"""Tests for market state normalization and event flow."""

from __future__ import annotations

import pytest

from alpha_engine.core.event_bus import EventBus
from alpha_engine.features.feature_engine import FeatureVector
from alpha_engine.state_engine.market_state_engine import (
    MarketStateEngine,
    minmax_scale,
    sigmoid,
)
from alpha_engine.state_engine.state_models import MarketStateVector


def _feature_vector(
    timestamp: int = 1700000000000,
    realized_vol: float = 0.2,
    trend_strength: float = 0.1,
    momentum: float = 0.02,
    range_compression: float = 0.9,
    return_zscore: float = 1.2,
) -> FeatureVector:
    return FeatureVector(
        timestamp=timestamp,
        realized_vol=realized_vol,
        atr=1.0,
        trend_strength=trend_strength,
        momentum=momentum,
        range_compression=range_compression,
        volume_spike=1.0,
        return_zscore=return_zscore,
    )


def test_normalization_helpers_bounds() -> None:
    assert 0.0 <= sigmoid(-1000.0) <= 1.0
    assert 0.0 <= sigmoid(0.0) <= 1.0
    assert 0.0 <= sigmoid(1000.0) <= 1.0
    assert minmax_scale(-1.0, 0.0, 1.0) == 0.0
    assert minmax_scale(2.0, 0.0, 1.0) == 1.0
    assert minmax_scale(0.5, 0.0, 1.0) == pytest.approx(0.5)


def test_market_state_generation_and_bounds() -> None:
    event_bus = EventBus()
    _ = MarketStateEngine(event_bus=event_bus)
    received: list[MarketStateVector] = []
    event_bus.subscribe("market.state", lambda state: received.append(state))

    event_bus.publish("market.features", _feature_vector())

    assert received, "expected market state event"
    latest = received[-1]
    assert latest.timestamp == 1700000000000
    assert 0.0 <= latest.trend_state <= 1.0
    assert 0.0 <= latest.volatility_state <= 1.0
    assert 0.0 <= latest.liquidity_stress <= 1.0
    assert 0.0 <= latest.momentum_pressure <= 1.0
    assert 0.0 <= latest.compression_state <= 1.0


def test_event_publishing() -> None:
    event_bus = EventBus()
    _ = MarketStateEngine(event_bus=event_bus)
    published_count = 0

    def _capture(_: MarketStateVector) -> None:
        nonlocal published_count
        published_count += 1

    event_bus.subscribe("market.state", _capture)

    event_bus.publish("market.features", _feature_vector(timestamp=1))
    event_bus.publish("market.features", _feature_vector(timestamp=2))

    assert published_count == 2


def test_correct_mapping_from_feature_vector() -> None:
    event_bus = EventBus()
    engine = MarketStateEngine(event_bus=event_bus)
    received: list[MarketStateVector] = []
    event_bus.subscribe("market.state", lambda state: received.append(state))

    first = _feature_vector(timestamp=1, realized_vol=0.1, trend_strength=0.25, momentum=0.01, range_compression=0.8, return_zscore=1.5)
    second = _feature_vector(timestamp=2, realized_vol=0.3, trend_strength=-0.25, momentum=0.03, range_compression=0.6, return_zscore=-2.0)

    event_bus.publish("market.features", first)
    event_bus.publish("market.features", second)

    latest = received[-1]
    expected_trend = (2.0 * sigmoid(abs(second.trend_strength) * engine.TREND_SIGMOID_STEEPNESS)) - 1.0
    rolling_baseline = (first.realized_vol + second.realized_vol) / 2.0
    expected_volatility = minmax_scale(second.realized_vol / rolling_baseline, engine.VOL_RATIO_MIN, engine.VOL_RATIO_MAX)
    expected_liquidity = minmax_scale(abs(second.return_zscore), 0.0, engine.RETURN_ZSCORE_MAX)
    expected_momentum = minmax_scale(second.momentum, engine.MOMENTUM_MIN, engine.MOMENTUM_MAX)
    expected_compression = 1.0 - minmax_scale(second.range_compression, engine.COMPRESSION_MIN, engine.COMPRESSION_MAX)

    assert latest.timestamp == second.timestamp
    assert latest.trend_state == pytest.approx(expected_trend)
    assert latest.volatility_state == pytest.approx(expected_volatility)
    assert latest.liquidity_stress == pytest.approx(expected_liquidity)
    assert latest.momentum_pressure == pytest.approx(expected_momentum)
    assert latest.compression_state == pytest.approx(expected_compression)
