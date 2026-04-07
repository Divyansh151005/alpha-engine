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
    # Phase 11 - raw features for alpha signals
    rolling_return_5: float = 0.0
    rolling_return_10: float = 0.0
    rolling_return_20: float = 0.0
    # Regime proxy: abs(rolling 20-bar return); ADX-like trend strength for filters.
    trend_strength: float = 0.0
    latest_bar_return: float = 0.0
    volatility_breakout: float = 1.0  # vol ratio: 1=normal, >1=expansion
    momentum_acceleration: float = 0.0
    mean_reversion_strength: float = 0.0  # z-score of price vs mean
    bollinger_band_distance: float = 0.0  # signed distance from middle band
    range_expansion: float = 1.0  # range ratio
    price_vs_recent_high_low: float = 0.5  # 0=at low, 1=at high
    ema_crossover_strength: float = 0.0
    # Last known perpetual funding (aligned to bar close); NaN if unavailable.
    funding_rate: float = float("nan")
    # Bar-to-bar relative OI change: (OI_t - OI_{t-1}) / OI_{t-1}; NaN if unavailable.
    oi_change: float = float("nan")
    # Spot / perp base symbol from the active candle (e.g. ``BTC/USDT``); None if unknown.
    symbol: str | None = None
