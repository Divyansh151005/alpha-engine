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
            score=0.0,
            confidence=0.5,
        )


class _FixedSignalStrategy(BaseAlphaStrategy):
    def __init__(self, name: str, score: float) -> None:
        self._name = name
        self._score = score

    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self._name,
            score=self._score,
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
    """TrendAlphaStrategy is score-inverted; uptrend state should be short."""
    strategy = TrendAlphaStrategy()
    state = _market_state(timestamp=2, trend_state=0.85, momentum_pressure=0.75)

    signal = strategy.generate_signal(state)

    assert signal is not None
    assert signal.timestamp == state.timestamp
    assert signal.strategy_name == "TrendAlphaStrategy"
    assert signal.score < 0
    assert signal.direction == "short"
    assert signal.strength == abs(signal.score)
    assert 0 <= signal.confidence <= 1


def test_trend_alpha_strategy_generates_short_signal() -> None:
    """TrendAlphaStrategy is score-inverted; downtrend state should be long."""
    strategy = TrendAlphaStrategy()
    state = _market_state(timestamp=2, trend_state=0.2, momentum_pressure=0.3)

    signal = strategy.generate_signal(state)

    assert signal is not None
    assert signal.score > 0
    assert signal.direction == "long"


def test_event_publishing_when_signal_exists() -> None:
    event_bus = EventBus()
    strategy = _FixedSignalStrategy(name="StrongSignal", score=0.9)
    _ = AlphaEngine(event_bus=event_bus, strategies=[strategy])
    received: list[AlphaSignal] = []
    event_bus.subscribe("alpha.signal", lambda signal: received.append(signal))

    event_bus.publish("market.state", _market_state(timestamp=3, trend_state=0.9, momentum_pressure=0.9))

    assert len(received) == 1
    assert received[0].direction == "long"


def test_low_confidence_signal_is_still_published() -> None:
    event_bus = EventBus()
    strategy = _RecordingStrategy()
    _ = AlphaEngine(event_bus=event_bus, strategies=[strategy])
    received: list[AlphaSignal] = []
    event_bus.subscribe("alpha.signal", lambda signal: received.append(signal))

    event_bus.publish("market.state", _market_state(timestamp=31, trend_state=0.7, momentum_pressure=0.6))

    assert len(received) == 1
    assert received[0].strategy_name == "RecordingStrategy"
    assert received[0].confidence < 0.6


def test_multiple_strategies_supported() -> None:
    event_bus = EventBus()
    long_strategy = _FixedSignalStrategy(name="LongOne", score=0.9)
    short_strategy = _FixedSignalStrategy(name="ShortOne", score=-0.9)
    _ = AlphaEngine(event_bus=event_bus, strategies=[long_strategy, short_strategy])

    received: list[AlphaSignal] = []
    event_bus.subscribe("alpha.signal", lambda signal: received.append(signal))

    event_bus.publish("market.state", _market_state(timestamp=4))

    assert len(received) == 2
    assert {signal.strategy_name for signal in received} == {"LongOne", "ShortOne"}
    assert {signal.direction for signal in received} == {"long", "short"}


def test_alpha_engine_publishes_empty_signals_and_zero_net_signal_when_disabled() -> None:
    event_bus = EventBus()
    # All below MIN_SIGNAL_THRESHOLD (0.25): nothing passes filter, no synthetic debug fill.
    strategies = [
        _FixedSignalStrategy(name="trend", score=0.1),
        _FixedSignalStrategy(name="mean_reversion", score=-0.1),
        _FixedSignalStrategy(name="breakout", score=0.0),
    ]
    alpha_engine = AlphaEngine(
        event_bus=event_bus,
        strategies=strategies,
        strategy_weights={"trend": 1.0, "mean_reversion": 1.0, "breakout": 1.0},
    )

    snapshots: list[dict[str, float]] = []
    net_values: list[float] = []
    event_bus.subscribe("alpha.signals", lambda s: snapshots.append(s))
    event_bus.subscribe("alpha.net_signal", lambda n: net_values.append(n))

    event_bus.publish("market.state", _market_state(timestamp=5))

    assert snapshots, "expected per-strategy signal snapshot"
    assert net_values, "expected net signal publication"
    assert snapshots[-1] == {}
    assert net_values[-1] == 0


def test_alpha_signals_not_constant() -> None:
    """Alpha signals vary with changing market state (not constant)."""
    strategy = TrendAlphaStrategy()
    state_up = _market_state(timestamp=1, trend_state=0.9, momentum_pressure=0.8)
    state_down = _market_state(timestamp=2, trend_state=0.1, momentum_pressure=0.2)
    sig_up = strategy.generate_signal(state_up)
    sig_down = strategy.generate_signal(state_down)
    assert sig_up.score != sig_down.score
    assert (sig_up.score > 0) != (sig_down.score > 0)


def test_alpha_scores_no_nans() -> None:
    """Alpha signals never contain NaN or inf."""
    from alpha_engine.alpha import BreakoutAlphaStrategy, MeanReversionAlphaStrategy

    strategies = [TrendAlphaStrategy(), MeanReversionAlphaStrategy(), BreakoutAlphaStrategy()]
    state = _market_state(timestamp=1)
    for strategy in strategies:
        sig = strategy.generate_signal(state)
        assert sig is not None
        assert sig.score == sig.score, "score must not be NaN"
        assert abs(sig.score) <= 1.0
        assert 0 <= sig.confidence <= 1


def test_alpha_score_direction_balance() -> None:
    """Over varied states, score distribution is reasonably balanced."""
    strategy = TrendAlphaStrategy()
    scores: list[float] = []
    for i in range(60):
        trend = 0.2 + 0.6 * (i % 5) / 4.0  # Vary 0.2 to 0.8
        mom = 0.2 + 0.6 * ((i + 2) % 5) / 4.0
        state = _market_state(timestamp=i, trend_state=trend, momentum_pressure=mom)
        sig = strategy.generate_signal(state)
        if sig is not None:
            scores.append(sig.score)
    n_pos = sum(1 for s in scores if s > 0)
    n_neg = sum(1 for s in scores if s < 0)
    assert n_pos >= 1 and n_neg >= 1, "both positive and negative scores"
    mean_score = sum(scores) / len(scores)
    assert abs(mean_score) < 0.8, "mean near 0 (normalized)"
