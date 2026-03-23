"""Risk engine for dynamic exposure reduction during drawdowns."""

from __future__ import annotations

from alpha_engine.allocation_engine.allocation_models import PositionAllocation
from alpha_engine.core.event_bus import EventBus
from alpha_engine.risk_engine.risk_models import RiskState

# Numerical stability
_PEAK_FLOOR = 1e-12


def _drawdown_factor(drawdown: float) -> float:
    """Compute risk scaling factor from drawdown percentage (0-1 scale).
    Deterministic: <5% -> 1.0, >=5% -> 0.8, >=10% -> 0.5, >=15% -> 0.2, >=20% -> 0.0
    """
    if drawdown < 0.05:
        return 1.0
    if drawdown < 0.10:
        return 0.8
    if drawdown < 0.15:
        return 0.5
    if drawdown < 0.20:
        return 0.2
    return 0.0


class RiskEngine:
    """Protect capital by scaling exposure during drawdowns."""

    def __init__(
        self,
        event_bus: EventBus,
        initial_equity: float = 100.0,
        equity_decay_per_update: float = 0.0,
        force_full_exposure: bool = False,
    ) -> None:
        self._event_bus = event_bus
        self._equity = initial_equity
        self._peak_equity = initial_equity
        self._equity_decay = max(0.0, min(1.0, equity_decay_per_update))
        self._force_full_exposure = force_full_exposure

        self._event_bus.subscribe("portfolio.position", self._handle_position)

    def _handle_position(self, positions: list[PositionAllocation]) -> None:
        """On each position update: mock equity, update peak, compute drawdown, scale, publish."""
        # 1. Mock equity update
        self._equity = self._equity * (1.0 - self._equity_decay)
        if self._equity < 0:
            self._equity = 0.0

        # 2. Update peak equity
        self._peak_equity = max(self._peak_equity, self._equity)

        # 3. Compute drawdown (safe: no division by zero)
        peak = max(self._peak_equity, _PEAK_FLOOR)
        drawdown = (peak - self._equity) / peak
        drawdown = max(0.0, min(1.0, drawdown))

        # 4. Get factor and apply risk scaling
        factor = 1.0 if self._force_full_exposure else _drawdown_factor(drawdown)
        final_positions: list[PositionAllocation] = []
        for pos in positions:
            final_fraction = pos.capital_fraction * factor
            final_fraction = max(0.0, min(1.0, final_fraction))
            final_positions.append(
                PositionAllocation(
                    timestamp=pos.timestamp,
                    strategy_name=pos.strategy_name,
                    direction=pos.direction,
                    capital_fraction=final_fraction,
                    risk_weight=pos.risk_weight,
                )
            )

        self._event_bus.publish("portfolio.final", final_positions)

    def get_risk_state(self) -> RiskState:
        """Return current risk state (for testing)."""
        peak = max(self._peak_equity, _PEAK_FLOOR)
        drawdown = (peak - self._equity) / peak if peak > 0 else 0.0
        drawdown = max(0.0, min(1.0, drawdown))
        return RiskState(
            equity=self._equity,
            peak_equity=self._peak_equity,
            drawdown=drawdown,
        )
