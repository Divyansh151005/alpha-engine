"""Tests for risk and drawdown engine."""

from __future__ import annotations

from alpha_engine.allocation_engine import PositionAllocation
from alpha_engine.core.event_bus import EventBus
from alpha_engine.risk_engine import RiskEngine


def _position(capital_fraction: float = 0.5) -> list[PositionAllocation]:
    return [
        PositionAllocation(
            timestamp=1000,
            strategy_name="test",
            direction="long",
            capital_fraction=capital_fraction,
            risk_weight=0.5,
        )
    ]


def test_drawdown_calculation() -> None:
    """Drawdown = (peak - current) / peak."""
    event_bus = EventBus()
    engine = RiskEngine(
        event_bus=event_bus,
        initial_equity=100.0,
        equity_decay_per_update=0.05,
    )
    event_bus.subscribe("portfolio.position", lambda _: None)

    event_bus.publish("portfolio.position", _position())

    state = engine.get_risk_state()
    assert state.equity == 95.0
    assert state.peak_equity == 100.0
    assert abs(state.drawdown - 0.05) < 1e-9


def test_scaling_behavior_no_drawdown() -> None:
    """Drawdown < 5%: factor = 1.0."""
    event_bus = EventBus()
    RiskEngine(
        event_bus=event_bus,
        initial_equity=100.0,
        equity_decay_per_update=0.04,
    )
    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.final", received.append)

    event_bus.publish("portfolio.position", _position(capital_fraction=0.6))

    assert len(received) == 1
    assert received[0][0].capital_fraction == 0.6


def test_scaling_behavior_5_percent() -> None:
    """Drawdown >= 5%: factor = 0.8."""
    event_bus = EventBus()
    RiskEngine(
        event_bus=event_bus,
        initial_equity=100.0,
        equity_decay_per_update=0.05,
    )
    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.final", received.append)

    event_bus.publish("portfolio.position", _position(capital_fraction=0.5))

    assert len(received) == 1
    assert abs(received[0][0].capital_fraction - 0.4) < 1e-9  # 0.5 * 0.8 = 0.4


def test_scaling_behavior_10_percent() -> None:
    """Drawdown >= 10%: factor = 0.5."""
    event_bus = EventBus()
    RiskEngine(
        event_bus=event_bus,
        initial_equity=100.0,
        equity_decay_per_update=0.10,
    )
    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.final", received.append)

    event_bus.publish("portfolio.position", _position(capital_fraction=0.6))

    assert len(received) == 1
    assert abs(received[0][0].capital_fraction - 0.3) < 1e-9  # 0.6 * 0.5 = 0.3


def test_scaling_behavior_15_percent() -> None:
    """Drawdown >= 15%: factor = 0.2."""
    event_bus = EventBus()
    RiskEngine(
        event_bus=event_bus,
        initial_equity=100.0,
        equity_decay_per_update=0.15,
    )
    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.final", received.append)

    event_bus.publish("portfolio.position", _position(capital_fraction=1.0))

    assert len(received) == 1
    assert abs(received[0][0].capital_fraction - 0.2) < 1e-9  # 1.0 * 0.2 = 0.2


def test_kill_switch_works() -> None:
    """Drawdown >= 20%: factor = 0.0, all positions zero."""
    event_bus = EventBus()
    RiskEngine(
        event_bus=event_bus,
        initial_equity=100.0,
        equity_decay_per_update=0.20,
    )
    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.final", received.append)

    event_bus.publish("portfolio.position", _position(capital_fraction=0.8))

    assert len(received) == 1
    assert received[0][0].capital_fraction == 0.0


def test_output_bounded() -> None:
    """Final allocations are non-negative and at most 1.0."""
    event_bus = EventBus()
    RiskEngine(
        event_bus=event_bus,
        initial_equity=100.0,
        equity_decay_per_update=0.10,
    )
    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.final", received.append)

    event_bus.publish(
        "portfolio.position",
        [
            PositionAllocation(
                timestamp=1000,
                strategy_name="a",
                direction="long",
                capital_fraction=1.5,  # Invalid input
                risk_weight=0.5,
            ),
            PositionAllocation(
                timestamp=1000,
                strategy_name="b",
                direction="short",
                capital_fraction=-0.1,  # Invalid input
                risk_weight=0.5,
            ),
        ],
    )

    assert len(received) == 1
    for pos in received[0]:
        assert 0.0 <= pos.capital_fraction <= 1.0


def test_deterministic_behavior() -> None:
    """Same inputs produce same outputs."""
    event_bus1 = EventBus()
    engine1 = RiskEngine(
        event_bus=event_bus1,
        initial_equity=100.0,
        equity_decay_per_update=0.07,
    )
    received1: list[list[PositionAllocation]] = []
    event_bus1.subscribe("portfolio.final", received1.append)

    event_bus2 = EventBus()
    engine2 = RiskEngine(
        event_bus=event_bus2,
        initial_equity=100.0,
        equity_decay_per_update=0.07,
    )
    received2: list[list[PositionAllocation]] = []
    event_bus2.subscribe("portfolio.final", received2.append)

    positions = _position(capital_fraction=0.5)
    event_bus1.publish("portfolio.position", positions)
    event_bus2.publish("portfolio.position", positions)

    assert received1[0][0].capital_fraction == received2[0][0].capital_fraction
    assert engine1.get_risk_state().drawdown == engine2.get_risk_state().drawdown


def test_safe_calculation_zero_peak() -> None:
    """Zero or near-zero peak does not cause division errors."""
    event_bus = EventBus()
    RiskEngine(
        event_bus=event_bus,
        initial_equity=0.0,
        equity_decay_per_update=0.0,
    )
    received: list[list[PositionAllocation]] = []
    event_bus.subscribe("portfolio.final", received.append)

    event_bus.publish("portfolio.position", _position(capital_fraction=0.5))

    assert len(received) == 1
    assert 0.0 <= received[0][0].capital_fraction <= 1.0
