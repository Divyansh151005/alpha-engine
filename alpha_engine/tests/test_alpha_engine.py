"""Tests for alpha engine orchestration and strategy modules."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from alpha_engine.alpha import AlphaEngine, AlphaSignal, BaseAlphaStrategy, TrendAlphaStrategy
from alpha_engine.core.event_bus import EventBus
from alpha_engine.state_engine.state_models import MarketStateVector


def _market_state(
    timestamp: int = 1700000000000,
    trend_state: float = 0.8,
    momentum_pressure: float = 0.7,
) -> MarketStateVector:
    return MarketStateVector(
        timestamp=timestamp,
        trend_state=trend_state,
        volatility_state=0.5,
        liquidity_stress=0.4,
        momentum_pressure=momentum_pressure,
        compression_state=0.6,
    )


@dataclass
class _RecordingStrategy(BaseAlphaStrategy):
    received_count: int = 0
    last_state: MarketStateVector | None = None

    def on_market_state(self, state: MarketStateVector) -> None:
        self.received_count += 1
        self.last_state = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name="RecordingStrategy",
            direction="flat",
            strength=0.5,
            confidence=0.5,
        )


class _FixedSignalStrategy(BaseAlphaStrategy):
    def __init__(self, name: str, direction: str) -> None:
        self._name = name
        self._direction = direction

    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self._name,
            direction=self._direction,
            strength=0.9,
            confidence=0.8,
        )


def test_strategy_receives_market_state() -> None:
    event_bus = EventBus()
    strategy = _RecordingStrategy()
    _ = AlphaEngine(event_bus=event_bus, strategies=[strategy])

    state = _market_state(timestamp=1)
    event_bus.publish("market.state", state)

    assert strategy.received_count == 1
    assert strategy.last_state == state


def test_trend_alpha_strategy_generates_long_signal() -> None:
    strategy = TrendAlphaStrategy()
    state = _market_state(timestamp=2, trend_state=0.85, momentum_pressure=0.75)

    signal = strategy.generate_signal(state)

    assert signal is not None
    assert signal.timestamp == state.timestamp
    assert signal.strategy_name == "TrendAlphaStrategy"
    assert signal.direction == "long"
    assert signal.strength == state.trend_state
    assert signal.confidence == state.momentum_pressure


def test_trend_alpha_strategy_generates_short_signal() -> None:
    """Trend produces short when downtrend + momentum confirming down."""
    strategy = TrendAlphaStrategy()
    state = _market_state(timestamp=2, trend_state=0.2, momentum_pressure=0.3)

    signal = strategy.generate_signal(state)

    assert signal is not None
    assert signal.direction == "short"
    assert signal.strength == pytest.approx(0.8)  # 1.0 - trend_state
    assert signal.confidence == pytest.approx(0.7)  # 1.0 - momentum_pressure


def test_event_publishing_when_signal_exists() -> None:
    event_bus = EventBus()
    strategy = TrendAlphaStrategy()
    _ = AlphaEngine(event_bus=event_bus, strategies=[strategy])
    received: list[AlphaSignal] = []
    event_bus.subscribe("alpha.signal", lambda signal: received.append(signal))

    event_bus.publish("market.state", _market_state(timestamp=3, trend_state=0.9, momentum_pressure=0.9))

    assert len(received) == 1
    assert received[0].direction == "long"


def test_multiple_strategies_supported() -> None:
    event_bus = EventBus()
    long_strategy = _FixedSignalStrategy(name="LongOne", direction="long")
    short_strategy = _FixedSignalStrategy(name="ShortOne", direction="short")
    _ = AlphaEngine(event_bus=event_bus, strategies=[long_strategy, short_strategy])

    received: list[AlphaSignal] = []
    event_bus.subscribe("alpha.signal", lambda signal: received.append(signal))

    event_bus.publish("market.state", _market_state(timestamp=4))

    assert len(received) == 2
    assert {signal.strategy_name for signal in received} == {"LongOne", "ShortOne"}
    assert {signal.direction for signal in received} == {"long", "short"}
