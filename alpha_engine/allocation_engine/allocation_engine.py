"""Capital allocator that emits portfolio allocations from alpha signals."""

from __future__ import annotations

import math

from alpha_engine.allocation_engine.allocation_models import PositionAllocation
from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.logger import get_logger
from alpha_engine.regime import RegimeState, RegimeType, get_regime_multipliers

_EPS = 1e-9
_logger = get_logger(__name__)

# Strategy name -> regime type for weight adjustment (trend, breakout, mr)
_STRATEGY_TO_REGIME_TYPE: dict[str, str] = {
    "TrendAlphaStrategy": "trend",
    "BreakoutAlphaStrategy": "breakout",
    "MeanReversionAlphaStrategy": "mr",
}


def _direction_sign(signal: AlphaSignal) -> float:
    """Use score for direction: positive=long, negative=short."""
    return max(-1.0, min(1.0, signal.score))


def _corr(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((y - mb) ** 2 for y in b) / (len(a) - 1)
    denom = math.sqrt(va * vb)
    if denom <= 1e-12:
        if va <= 1e-12 and vb <= 1e-12:
            return 1.0 if (ma * mb >= 0) else -1.0
        return 0.0
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) - 1)
    return cov / denom


class CapitalAllocator:
    """Allocate capital across buffered strategy signals on each signal update."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe("alpha.signal", self.on_signal)
        self._event_bus.subscribe("market.regime", self._handle_regime)
        # Public buffer for diagnostics and external inspection.
        self.signals: dict[str, AlphaSignal] = {}
        self._latest_regime: RegimeType | None = None

    def _handle_regime(self, regime_state: RegimeState) -> None:
        """Store latest regime for weight adjustment."""
        self._latest_regime = regime_state.regime

    def on_signal(self, signal: AlphaSignal) -> None:
        """Buffer latest signal and emit updated allocation snapshot."""
        self.signals[signal.strategy_name] = signal
        self._recompute_and_publish()

    def _recompute_and_publish(self) -> None:
        if not self.signals:
            return

        allocations = self._compute_allocations()
        _logger.info(
            "allocation.generated count=%s strategies=%s",
            len(allocations),
            [a.strategy_name for a in allocations],
        )
        self._event_bus.publish("portfolio.allocation", allocations)

    def _compute_allocations(self) -> list[PositionAllocation]:
        if not self.signals:
            return []

        signals = list(self.signals.values())
        if not signals:
            return []
        weights = [max(_EPS, abs(s.score)) for s in signals]

        total = sum(weights)
        if total <= 0:
            total = 1.0

        allocations: list[PositionAllocation] = []
        for sig, weight in sorted(zip(signals, weights), key=lambda x: x[0].strategy_name):
            capital_fraction = weight / total
            capital_fraction = max(0.0, min(1.0, capital_fraction))
            risk_weight = max(0.0, min(1.0, sig.confidence))
            allocations.append(
                PositionAllocation(
                    timestamp=sig.timestamp,
                    strategy_name=sig.strategy_name,
                    direction="long" if sig.score > 0 else "short",
                    capital_fraction=capital_fraction,
                    risk_weight=risk_weight,
                )
            )

        # Regime-based weight adjustment
        allocations = self._apply_regime_weights(allocations)

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

        return allocations

    def _apply_regime_weights(
        self, allocations: list[PositionAllocation]
    ) -> list[PositionAllocation]:
        """Apply regime-based multipliers to capital fractions and re-normalize."""
        regime = self._latest_regime
        if regime is None:
            return allocations
        multipliers = get_regime_multipliers(regime)

        adjusted: list[PositionAllocation] = []
        for a in allocations:
            regime_type = _STRATEGY_TO_REGIME_TYPE.get(a.strategy_name, "trend")
            mult = multipliers.get(regime_type, 1.0)
            new_frac = max(0.0, min(1.0, a.capital_fraction * mult))
            adjusted.append(
                PositionAllocation(
                    timestamp=a.timestamp,
                    strategy_name=a.strategy_name,
                    direction=a.direction,
                    capital_fraction=new_frac,
                    risk_weight=a.risk_weight,
                )
            )
            _logger.info(
                "allocation.regime strategy=%s regime=%s mult=%.2f orig=%.4f adj=%.4f",
                a.strategy_name,
                regime.value,
                mult,
                a.capital_fraction,
                new_frac,
            )

        total = sum(x.capital_fraction for x in adjusted)
        if total <= _EPS:
            return allocations
        scale = 1.0 / total
        return [
            PositionAllocation(
                timestamp=a.timestamp,
                strategy_name=a.strategy_name,
                direction=a.direction,
                capital_fraction=a.capital_fraction * scale,
                risk_weight=a.risk_weight,
            )
            for a in adjusted
        ]
