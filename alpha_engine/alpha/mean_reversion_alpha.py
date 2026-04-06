"""Mean reversion alpha from upstream return z-score."""

from __future__ import annotations

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.state_engine.state_models import MarketStateVector

class MeanReversionAlphaStrategy(BaseAlphaStrategy):
    """Thresholded mean reversion using precomputed return z-score."""

    def __init__(self) -> None:
        self._z_values: list[float] = []

    def on_market_state(self, state: MarketStateVector) -> None:
        pass

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal:
        print(
            "[MR INPUT] "
            f"z={state.return_zscore}"
        )
        if abs(state.return_zscore) > 1.5:
            print(
                "[MR HIGH Z DETECTED] "
                f"z={state.return_zscore}"
            )
        z = state.return_zscore
        self._z_values.append(z)
        if len(self._z_values) % 100 == 0:
            import numpy as np

            arr = np.array(self._z_values)
            print(
                "[Z DISTRIBUTION] "
                f"count={len(arr)} "
                f"mean={arr.mean():.4f} "
                f"std={arr.std():.4f} "
                f"min={arr.min():.4f} "
                f"max={arr.max():.4f}"
            )
        if z > 2.0:
            signal = -1.0
        elif z < -2.0:
            signal = 1.0
        else:
            signal = 0.0

        print(
            "[MR SIGNAL] "
            f"z={z} "
            f"signal={signal}"
        )

        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self.__class__.__name__,
            score=signal,
            confidence=1.0,
        )
