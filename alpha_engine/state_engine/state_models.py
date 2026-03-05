"""State model definitions for normalized market conditions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketStateVector:
    """Normalized market state values in the [0, 1] range."""

    timestamp: int
    trend_state: float
    volatility_state: float
    liquidity_stress: float
    momentum_pressure: float
    compression_state: float
