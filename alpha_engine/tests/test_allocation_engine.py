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
