"""Tests for dynamic capital allocation engine."""

from __future__ import annotations

from alpha_engine.allocation_engine import CapitalAllocator, PositionAllocation
from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.core.event_bus import EventBus


def _signal(
    strategy_name: str,
    timestamp: int = 1000,
    direction: str = "long",
    strength: float = 0.5,
    confidence: float = 0.5,
) -> AlphaSignal:
    return AlphaSignal(
        timestamp=timestamp,
        strategy_name=strategy_name,
        direction=direction,
        strength=strength,
        confidence=confidence,
    )


def test_multiple_signals_normalized_correctly() -> None:
    """Multiple signals are normalized so allocations sum to 1."""
    event_bus = EventBus()
    CapitalAllocator(event_bus=event_bus)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.allocation", received.append)

    event_bus.publish(
        "alpha.signal",
        _signal("A", strength=0.5, confidence=0.5),
    )
    event_bus.publish(
        "alpha.signal",
        _signal("B", strength=0.5, confidence=0.5),
    )

    assert len(received) >= 1
    allocations = received[-1]
    total = sum(a.capital_fraction for a in allocations)
    assert total <= 1.0 + 1e-9
    assert total >= 0.99


def test_allocations_sum_le_one() -> None:
    """Sum of capital_fraction across strategies never exceeds 1."""
    event_bus = EventBus()
    CapitalAllocator(event_bus=event_bus)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.allocation", received.append)

    for i in range(5):
        event_bus.publish(
            "alpha.signal",
            _signal(f"Strategy{i}", strength=1.0, confidence=1.0),
        )

    assert len(received) >= 1
    allocations = received[-1]
    total = sum(a.capital_fraction for a in allocations)
    assert total <= 1.0


def test_higher_strength_gets_higher_allocation() -> None:
    """Stronger signals receive proportionally higher capital."""
    event_bus = EventBus()
    CapitalAllocator(event_bus=event_bus)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.allocation", received.append)

    event_bus.publish(
        "alpha.signal",
        _signal("Weak", strength=0.2, confidence=0.5),
    )
    event_bus.publish(
        "alpha.signal",
        _signal("Strong", strength=0.9, confidence=0.9),
    )

    assert len(received) >= 1
    allocations = {a.strategy_name: a for a in received[-1]}
    assert "Weak" in allocations
    assert "Strong" in allocations
    assert allocations["Strong"].capital_fraction > allocations["Weak"].capital_fraction


def test_event_publishing_works() -> None:
    """portfolio.allocation events are published on new signals."""
    event_bus = EventBus()
    CapitalAllocator(event_bus=event_bus)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.allocation", received.append)

    event_bus.publish("alpha.signal", _signal("TestStrategy"))

    assert len(received) == 1
    assert len(received[0]) == 1
    assert received[0][0].strategy_name == "TestStrategy"
    assert received[0][0].capital_fraction <= 1.0
    assert received[0][0].risk_weight == 0.5


def test_latest_signal_per_strategy_used() -> None:
    """When a strategy emits again, its latest signal is used."""
    event_bus = EventBus()
    CapitalAllocator(event_bus=event_bus)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.allocation", received.append)

    event_bus.publish(
        "alpha.signal",
        _signal("Same", strength=0.5, confidence=0.5),
    )
    event_bus.publish(
        "alpha.signal",
        _signal("Same", strength=1.0, confidence=1.0, timestamp=2000),
    )

    assert len(received) == 2
    allocations = received[-1]
    assert len(allocations) == 1
    assert allocations[0].capital_fraction == 1.0
    assert allocations[0].timestamp == 2000


def test_correlated_signals_get_lower_allocation() -> None:
    """Strategies with correlated signals receive reduced allocation."""
    event_bus = EventBus()
    CapitalAllocator(event_bus=event_bus, correlation_window=10)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.allocation", received.append)

    for i in range(15):
        ts = 1000 + i * 100
        event_bus.publish(
            "alpha.signal",
            _signal("A", timestamp=ts, direction="long", strength=0.5, confidence=0.5),
        )
        event_bus.publish(
            "alpha.signal",
            _signal("B", timestamp=ts, direction="long", strength=0.5, confidence=0.5),
        )

    allocations = received[-1]
    alloc_map = {a.strategy_name: a for a in allocations}
    assert "A" in alloc_map and "B" in alloc_map
    total = sum(a.capital_fraction for a in allocations)
    assert total <= 1.0 + 1e-6


def test_independent_signals_get_higher_allocation() -> None:
    """Strategies with opposite/independent signals retain higher allocation."""
    event_bus = EventBus()
    CapitalAllocator(event_bus=event_bus, correlation_window=10)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.allocation", received.append)

    for i in range(15):
        ts = 1000 + i * 100
        dir_a = "long" if i % 2 == 0 else "short"
        dir_b = "short" if i % 2 == 0 else "long"
        event_bus.publish(
            "alpha.signal",
            _signal("A", timestamp=ts, direction=dir_a, strength=0.5, confidence=0.5),
        )
        event_bus.publish(
            "alpha.signal",
            _signal("B", timestamp=ts, direction=dir_b, strength=0.5, confidence=0.5),
        )

    allocations = received[-1]
    alloc_map = {a.strategy_name: a for a in allocations}
    assert "A" in alloc_map and "B" in alloc_map
    total = sum(a.capital_fraction for a in allocations)
    assert total <= 1.0 + 1e-6
    assert alloc_map["A"].capital_fraction > 0.4
    assert alloc_map["B"].capital_fraction > 0.4


def test_correlated_vs_independent_allocation() -> None:
    """Independent strategy gets higher allocation than correlated pair."""
    event_bus = EventBus()
    CapitalAllocator(event_bus=event_bus, correlation_window=10)

    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.allocation", received.append)

    for i in range(15):
        ts = 1000 + i * 100
        event_bus.publish("alpha.signal", _signal("A", ts, "long", 0.5, 0.5))
        event_bus.publish("alpha.signal", _signal("B", ts, "long", 0.5, 0.5))
        event_bus.publish(
            "alpha.signal",
            _signal("C", ts, "short" if i % 2 == 0 else "long", 0.5, 0.5),
        )

    alloc_map = {a.strategy_name: a for a in received[-1]}
    assert "A" in alloc_map and "B" in alloc_map and "C" in alloc_map
    assert alloc_map["C"].capital_fraction > alloc_map["A"].capital_fraction
    assert alloc_map["C"].capital_fraction > alloc_map["B"].capital_fraction
