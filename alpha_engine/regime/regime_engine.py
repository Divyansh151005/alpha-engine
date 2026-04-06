"""Regime classification engine - detects market regime and publishes for weight adjustment."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.logger import get_logger
from alpha_engine.data.models import OHLCV
from alpha_engine.features.feature_engine import FeatureVector
from alpha_engine.features.volatility import calculate_returns

_logger = get_logger(__name__)

_EPS = 1e-12


class RegimeType(str, Enum):
    """Market regime classification."""

    TREND = "TREND"
    MEAN_REVERSION = "MEAN_REVERSION"
    HIGH_VOL = "HIGH_VOL"
    LOW_VOL = "LOW_VOL"


@dataclass
class RegimeState:
    """Current regime and diagnostic values."""

    regime: RegimeType
    timestamp: int
    trend_strength: float
    volatility: float
    return_autocorrelation: float
    vol_ratio: float


def get_regime_multipliers(regime: RegimeType) -> dict[str, float]:
    """
    Strategy-type multipliers per regime.
    Keys: trend, breakout, mr (mean reversion).
    TREND: boost trend + breakout, reduce MR.
    MEAN_REVERSION: boost MR, reduce trend.
    HIGH_VOL: reduce all positions.
    LOW_VOL: reduce breakout.
    """
    multipliers = {
        RegimeType.TREND: {"trend": 1.3, "breakout": 1.2, "mr": 0.6},
        RegimeType.MEAN_REVERSION: {"trend": 0.6, "breakout": 0.8, "mr": 1.3},
        RegimeType.HIGH_VOL: {"trend": 0.7, "breakout": 0.7, "mr": 0.7},
        RegimeType.LOW_VOL: {"trend": 1.0, "breakout": 0.6, "mr": 1.0},
    }
    return multipliers.get(regime, {"trend": 1.0, "breakout": 1.0, "mr": 1.0})


def _lag1_autocorrelation(returns: list[float]) -> float:
    """Lag-1 autocorrelation of returns. Negative => mean reverting."""
    if len(returns) < 3:
        return 0.0
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / max(n - 1, 1)
    if var < _EPS:
        return 0.0
    cov = sum((returns[i] - mean) * (returns[i - 1] - mean) for i in range(1, n)) / max(
        n - 2, 1
    )
    return cov / var


class RegimeEngine:
    """
    Detect market regime from trend_strength, volatility, and return autocorrelation.
    Publishes 'market.regime' for strategy weight adjustment.
    """

    TREND_STRENGTH_THRESHOLD = 0.02
    VOL_RATIO_HIGH_THRESHOLD = 1.5
    RETURN_AUTOCORR_WINDOW = 50
    VOL_BASELINE_WINDOW = 50

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._closes: deque[float] = deque(maxlen=self.RETURN_AUTOCORR_WINDOW + 2)
        self._rolling_vol: deque[float] = deque(maxlen=self.VOL_BASELINE_WINDOW)
        self._latest_regime: RegimeState | None = None

        self._event_bus.subscribe("market.features", self._on_features)
        self._event_bus.subscribe("market.candle", self._on_candle)

    def _on_candle(self, candle: OHLCV) -> None:
        """Accumulate close prices for return autocorrelation."""
        if math.isfinite(candle.close) and candle.close > 0:
            self._closes.append(float(candle.close))

    def _on_features(self, feature_vector: FeatureVector) -> None:
        """Detect regime from features and publish market.regime."""
        trend_strength = self._sanitize(feature_vector.trend_strength)
        volatility = max(0.0, self._sanitize(feature_vector.realized_vol))
        self._rolling_vol.append(volatility)

        # Return autocorrelation from closes
        returns = self._compute_returns()
        autocorr = _lag1_autocorrelation(returns) if len(returns) >= 3 else 0.0

        # Vol ratio for "high vol" detection
        baseline_vol = self._rolling_volatility_baseline()
        vol_ratio = volatility / max(baseline_vol, _EPS)

        regime = self._detect_regime(
            trend_strength=trend_strength,
            vol_ratio=vol_ratio,
            autocorr=autocorr,
        )
        regime_state = RegimeState(
            regime=regime,
            timestamp=feature_vector.timestamp,
            trend_strength=trend_strength,
            volatility=volatility,
            return_autocorrelation=autocorr,
            vol_ratio=vol_ratio,
        )
        self._latest_regime = regime_state
        _logger.info(
            "regime.detected regime=%s timestamp=%s trend_strength=%.4f vol_ratio=%.4f autocorr=%.4f",
            regime.value,
            feature_vector.timestamp,
            trend_strength,
            vol_ratio,
            autocorr,
        )
        self._event_bus.publish("market.regime", regime_state)

    def _compute_returns(self) -> list[float]:
        """Compute returns from close history."""
        if len(self._closes) < 2:
            return []
        arr = np.asarray(list(self._closes), dtype=np.float64)
        rets = calculate_returns(arr)
        return [float(r) for r in rets if math.isfinite(r)]

    def _rolling_volatility_baseline(self) -> float:
        if not self._rolling_vol:
            return _EPS
        return max(sum(self._rolling_vol) / len(self._rolling_vol), _EPS)

    def _detect_regime(
        self,
        trend_strength: float,
        vol_ratio: float,
        autocorr: float,
    ) -> RegimeType:
        """
        Priority order:
        1. If trend_strength high -> TREND
        2. If volatility high -> HIGH_VOL
        3. If autocorrelation negative -> MEAN_REVERSION
        4. Else -> LOW_VOL
        """
        if abs(trend_strength) >= self.TREND_STRENGTH_THRESHOLD:
            return RegimeType.TREND
        if vol_ratio >= self.VOL_RATIO_HIGH_THRESHOLD:
            return RegimeType.HIGH_VOL
        if autocorr < 0:
            return RegimeType.MEAN_REVERSION
        return RegimeType.LOW_VOL

    def _sanitize(self, value: float) -> float:
        return float(value) if math.isfinite(value) else 0.0

    def get_regime_multipliers(self, regime: RegimeType) -> dict[str, float]:
        """Delegate to module-level function."""
        return get_regime_multipliers(regime)

    def get_latest_regime(self) -> RegimeState | None:
        """Return last published regime state (for testing)."""
        return self._latest_regime
