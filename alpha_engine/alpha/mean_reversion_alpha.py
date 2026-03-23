"""Mean reversion alpha strategy - zscore + wick + volume confluence for strong signals."""

from __future__ import annotations

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.state_engine.state_models import MarketStateVector

ZSCORE_THRESHOLD = 2.0
WICK_THRESHOLD = 0.25  # Lower/upper wick must be >= 25% of range
VOLUME_SPIKE_THRESHOLD = 1.15  # Volume >= 15% above average


class MeanReversionAlphaStrategy(BaseAlphaStrategy):
    """Mean reversion: confluence of large return, wick, volume. Triggers less often, stronger signals."""

    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        z = state.return_zscore
        vol_spike = state.volume_spike
        lower_wick = state.lower_wick_ratio
        upper_wick = state.upper_wick_ratio

        # STRONG LONG: large negative return + long lower wick + volume spike
        if (
            z < -ZSCORE_THRESHOLD
            and lower_wick >= WICK_THRESHOLD
            and vol_spike >= VOLUME_SPIKE_THRESHOLD
        ):
            direction = "long"
        # STRONG SHORT: large positive return + long upper wick + volume spike
        elif (
            z > ZSCORE_THRESHOLD
            and upper_wick >= WICK_THRESHOLD
            and vol_spike >= VOLUME_SPIKE_THRESHOLD
        ):
            direction = "short"
        else:
            return None

        # Strength: combine zscore + wick + volume_spike, normalize
        z_component = min(abs(z) / 4.0, 1.0)
        wick_component = lower_wick if direction == "long" else upper_wick
        vol_component = min((vol_spike - 1.0) / 1.0, 1.0) if vol_spike > 1.0 else 0.0
        raw_strength = (z_component + wick_component + max(0.0, vol_component)) / 3.0
        strength = max(0.2, min(1.0, raw_strength))

        # Confidence: higher when volatility high and volume high
        vol_conf = state.volatility_state
        vol_spike_conf = min(vol_spike / 2.0, 1.0) if vol_spike > 1.0 else 0.5
        confidence = max(0.3, min(1.0, 0.5 * vol_conf + 0.5 * vol_spike_conf))

        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self.__class__.__name__,
            direction=direction,
            strength=strength,
            confidence=confidence,
        )
