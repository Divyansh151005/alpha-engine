"""Volatility targeting engine for consistent risk exposure."""

from __future__ import annotations

import math

from alpha_engine.allocation_engine.allocation_models import PositionAllocation
from alpha_engine.core.event_bus import EventBus
from alpha_engine.features.feature_engine import FeatureVector

# Numerical stability constants
_MIN_VOL_FLOOR = 1e-9
_MAX_SCALE = 10.0


class VolatilityTargetingEngine:
    """Adjust position sizes dynamically based on market volatility."""

    def __init__(self, event_bus: EventBus, target_vol: float) -> None:
        self._event_bus = event_bus
        self._target_vol = target_vol
        self._latest_realized_vol: float | None = None

        self._event_bus.subscribe("portfolio.allocation", self._handle_allocation)
        self._event_bus.subscribe("market.features", self._handle_features)

    def _handle_features(self, feature_vector: FeatureVector) -> None:
        """Store latest realized volatility from market features."""
        vol = feature_vector.realized_vol
        if vol is not None and math.isfinite(vol) and vol >= 0:
            self._latest_realized_vol = vol

    def _handle_allocation(self, allocations: list[PositionAllocation]) -> None:
        """Adjust allocations by volatility and publish to portfolio.position."""
        current_vol = self._latest_realized_vol

        if current_vol is None or current_vol <= 0:
            scale = 1.0
        else:
            safe_vol = max(current_vol, _MIN_VOL_FLOOR)
            raw_scale = self._target_vol / safe_vol
            scale = min(raw_scale, _MAX_SCALE)

        adjusted: list[PositionAllocation] = []
        for a in allocations:
            adjusted_fraction = a.capital_fraction * scale
            adjusted_fraction = max(0.0, min(1.0, adjusted_fraction))
            adjusted.append(
                PositionAllocation(
                    timestamp=a.timestamp,
                    strategy_name=a.strategy_name,
                    direction=a.direction,
                    capital_fraction=adjusted_fraction,
                    risk_weight=a.risk_weight,
                )
            )

        self._event_bus.publish("portfolio.position", adjusted)
