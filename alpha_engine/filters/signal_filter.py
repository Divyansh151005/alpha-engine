"""Signal threshold filter to eliminate weak signals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alpha_engine.core.logger import get_logger

if TYPE_CHECKING:
    from alpha_engine.state_engine.state_models import MarketStateVector

_logger = get_logger(__name__)

_BASE_THRESHOLD = 0.4
_HIGH_VOL_THRESHOLD = 0.3
_LOW_VOL_THRESHOLD = 0.5
_VOL_HIGH = 0.6  # volatility_state >= this -> high vol
_VOL_LOW = 0.4   # volatility_state <= this -> low vol


class SignalFilter:
    """
    Filter weak signals based on threshold.
    - If |signal| < threshold -> return 0 (no trade)
    - Else -> return signal

    Threshold adapts to volatility:
    - Base: 0.4
    - High volatility -> 0.3 (allow more signals)
    - Low volatility -> 0.5 (stricter)
    """

    def __init__(
        self,
        base_threshold: float = _BASE_THRESHOLD,
        high_vol_threshold: float = _HIGH_VOL_THRESHOLD,
        low_vol_threshold: float = _LOW_VOL_THRESHOLD,
    ) -> None:
        self._base_threshold = base_threshold
        self._high_vol_threshold = high_vol_threshold
        self._low_vol_threshold = low_vol_threshold

    def filter(self, strategy_name: str, signal: float, market_state: MarketStateVector) -> float:
        """
        Filter signal by threshold. Returns 0 if below threshold, else returns signal.

        Args:
            strategy_name: Strategy identifier (for logging).
            signal: Raw signal in [-1, 1].
            market_state: Market state for volatility-aware threshold.

        Returns:
            Filtered signal (0 or original signal).
        """
        threshold = self._get_threshold(market_state)
        if abs(signal) < threshold:
            _logger.info(
                "filter.blocked reason=low_signal strategy=%s signal=%.4f threshold=%.4f",
                strategy_name,
                signal,
                threshold,
            )
            return 0.0
        return signal

    def _get_threshold(self, market_state: MarketStateVector) -> float:
        """Compute volatility-adjusted threshold."""
        vol_state = getattr(market_state, "volatility_state", 0.5)
        if vol_state >= _VOL_HIGH:
            return self._high_vol_threshold
        if vol_state <= _VOL_LOW:
            return self._low_vol_threshold
        return self._base_threshold
