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
    return_zscore: float = 0.0  # Raw signed zscore for mean reversion
    lower_wick_ratio: float = 0.0  # Lower wick / range [0, 1]
    upper_wick_ratio: float = 0.0  # Upper wick / range [0, 1]
    volume_spike: float = 1.0  # Latest vol / rolling avg (1 = normal)
