"""Dynamic capital allocator based on alpha signal strength and confidence."""

from __future__ import annotations

from alpha_engine.allocation_engine.allocation_models import PositionAllocation
from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.core.event_bus import EventBus

_EPS = 1e-9


class CapitalAllocator:
    """Allocate capital across strategies based on signal strength and confidence."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe("alpha.signal", self._handle_signal)
        self._active_signals: dict[str, AlphaSignal] = {}

    def _handle_signal(self, signal: AlphaSignal) -> None:
        self._active_signals[signal.strategy_name] = signal
        self._recompute_and_publish()

    def _recompute_and_publish(self) -> None:
        if not self._active_signals:
            return

        allocations = self._compute_allocations()
        self._event_bus.publish("portfolio.allocation", allocations)

    def _compute_allocations(self) -> list[PositionAllocation]:
        signals = list(self._active_signals.values())
        if not signals:
            return []

        weights = [
            max(0.0, s.strength * s.confidence) + _EPS
            for s in signals
        ]
        total = sum(weights)
        if total <= 0:
            total = 1.0

        allocations: list[PositionAllocation] = []
        for sig, raw_alloc in zip(signals, weights):
            capital_fraction = raw_alloc / total
            capital_fraction = max(0.0, min(1.0, capital_fraction))
            risk_weight = max(0.0, min(1.0, sig.confidence))
            allocations.append(
                PositionAllocation(
                    timestamp=sig.timestamp,
                    strategy_name=sig.strategy_name,
                    direction=sig.direction,
                    capital_fraction=capital_fraction,
                    risk_weight=risk_weight,
                )
            )

        total_alloc = sum(a.capital_fraction for a in allocations)
        if total_alloc > 1.0 + _EPS:
            scale = 1.0 / total_alloc
            allocations = [
                PositionAllocation(
                    timestamp=a.timestamp,
                    strategy_name=a.strategy_name,
                    direction=a.direction,
                    capital_fraction=a.capital_fraction * scale,
                    risk_weight=a.risk_weight,
                )
                for a in allocations
            ]

        return sorted(allocations, key=lambda a: a.strategy_name)
