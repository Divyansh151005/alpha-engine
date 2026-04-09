"""Event-driven market state normalization engine."""

from __future__ import annotations

import math
from collections import deque
from math import exp, isfinite, tanh

import numpy as np

from alpha_engine.core.event_bus import EventBus
from alpha_engine.features.feature_engine import FeatureVector
from alpha_engine.state_engine.state_models import MarketStateVector

_EPS = 1e-12


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid output in [0, 1]."""
    if x >= 0.0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def minmax_scale(x: float, min_val: float, max_val: float) -> float:
    """Scale a value to [0, 1] with clipping."""
    if max_val <= min_val:
        return 0.0
    scaled = (x - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, float(scaled)))


def _rolling_std(values: deque[float]) -> float:
    """Rolling standard deviation."""
    if len(values) < 2:
        return _EPS
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return max(math.sqrt(max(var, 0.0)), _EPS)


def _rolling_mean(values: deque[float], k: int) -> float:
    """Mean of last k values."""
    if k <= 0 or len(values) < k:
        return 0.0
    return sum(list(values)[-k:]) / k


class MarketStateEngine:
    """Convert raw feature vectors into normalized market state vectors."""

    VOLATILITY_BASELINE_WINDOW = 50
    VOLATILITY_HISTORY_WINDOW = 200
    VOLATILITY_HIGH_PCT = 60.0
    VOLATILITY_MIN_HISTORY = 50
    TREND_SIGMOID_STEEPNESS = 8.0
    VOL_RATIO_MIN = 0.5
    VOL_RATIO_MAX = 2.0
    RETURN_ZSCORE_MAX = 3.0
    MOMENTUM_MIN = -0.05
    MOMENTUM_MAX = 0.05
    COMPRESSION_MIN = 0.5
    COMPRESSION_MAX = 1.5

    # Alpha amplification (Phase 9.2)
    AMPLIFY_ROLLING_WINDOW = 50
    AMPLIFY_TANH_K = 3.0
    AMPLIFY_MOMENTUM_SHORT = 5
    AMPLIFY_MOMENTUM_LONG = 20
    AMPLIFY_MOMENTUM_BOOST = 0.3
    AMPLIFY_VOL_THRESHOLD = 0.5
    AMPLIFY_LOW_VOL_REDUCE = 0.5

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._rolling_realized_vol: deque[float] = deque(maxlen=self.VOLATILITY_BASELINE_WINDOW)
        self._rolling_trend_raw: deque[float] = deque(maxlen=self.AMPLIFY_ROLLING_WINDOW)
        self._volatility_history: deque[float] = deque(maxlen=self.VOLATILITY_HISTORY_WINDOW)
        self._event_bus.subscribe("market.features", self.handle_features)

    @staticmethod
    def is_high_volatility(
        volatility: float,
        history: deque[float] | list[float] | np.ndarray,
        *,
        percentile: float = VOLATILITY_HIGH_PCT,
        min_history: int = VOLATILITY_MIN_HISTORY,
    ) -> bool:
        if len(history) < int(min_history):
            return True  # warmup phase
        values = np.asarray(list(history), dtype=np.float64)
        thr = float(np.percentile(values, float(percentile)))
        return float(volatility) > thr

    def handle_features(self, feature_vector: FeatureVector) -> None:
        """Map a feature vector into a stable, normalized market state vector."""
        realized_vol = self._sanitize_nonnegative(feature_vector.realized_vol)
        volatility = self._sanitize_nonnegative(getattr(feature_vector, "volatility", 0.0))
        self._rolling_realized_vol.append(realized_vol)
        self._volatility_history.append(volatility)
        rolling_vol = self._rolling_volatility_baseline()
        high_volatility = self.is_high_volatility(volatility, self._volatility_history)

        trend_state = self._amplified_trend_state(
            feature_vector.trend_strength,
            feature_vector.momentum,
            realized_vol,
            rolling_vol,
        )
        volatility_state = self._volatility_state(realized_vol, rolling_vol)
        liquidity_stress = minmax_scale(
            abs(self._sanitize(feature_vector.return_zscore)),
            0.0,
            self.RETURN_ZSCORE_MAX,
        )
        momentum_pressure = minmax_scale(
            self._sanitize(feature_vector.momentum),
            self.MOMENTUM_MIN,
            self.MOMENTUM_MAX,
        )
        compression_raw = minmax_scale(
            self._sanitize_nonnegative(feature_vector.range_compression),
            self.COMPRESSION_MIN,
            self.COMPRESSION_MAX,
        )
        compression_state = 1.0 - compression_raw

        return_zscore_raw = self._sanitize(feature_vector.return_zscore)
        low_wick = max(0.0, min(1.0, self._sanitize_nonnegative(feature_vector.lower_wick_ratio)))
        up_wick = max(0.0, min(1.0, self._sanitize_nonnegative(feature_vector.upper_wick_ratio)))
        vol_spike = max(0.0, self._sanitize_nonnegative(feature_vector.volume_spike))

        # Phase 11 - pass through raw features for alpha signals
        rolling_5 = self._sanitize(feature_vector.rolling_return_5)
        rolling_10 = self._sanitize(feature_vector.rolling_return_10)
        rolling_20 = self._sanitize(feature_vector.rolling_return_20)
        trend_strength_regime = max(
            0.0, self._sanitize_nonnegative(feature_vector.regime_trend_strength)
        )
        latest_bar_ret = self._sanitize(feature_vector.latest_bar_return)
        vol_breakout = max(0.0, self._sanitize_nonnegative(feature_vector.volatility_breakout))
        mom_accel = self._sanitize(feature_vector.momentum_acceleration)
        mr_strength = self._sanitize(feature_vector.mean_reversion_strength)
        bb_dist = self._sanitize(feature_vector.bollinger_band_distance)
        range_exp = max(0.0, self._sanitize_nonnegative(feature_vector.range_expansion))
        price_vs_range = max(0.0, min(1.0, self._sanitize(feature_vector.price_vs_recent_high_low)))
        ema_cross = self._sanitize(feature_vector.ema_crossover_strength)
        fund_r = float(feature_vector.funding_rate)
        if not isfinite(fund_r):
            fund_r = float("nan")
        oi_ch = float(feature_vector.oi_change)
        if not isfinite(oi_ch):
            oi_ch = float("nan")

        market_state_vector = MarketStateVector(
            timestamp=feature_vector.timestamp,
            trend_state=trend_state,
            volatility_state=volatility_state,
            volatility=volatility,
            liquidity_stress=liquidity_stress,
            momentum_pressure=momentum_pressure,
            compression_state=compression_state,
            high_volatility=high_volatility,
            return_zscore=return_zscore_raw,
            lower_wick_ratio=low_wick,
            upper_wick_ratio=up_wick,
            volume_spike=vol_spike,
            rolling_return_5=rolling_5,
            rolling_return_10=rolling_10,
            rolling_return_20=rolling_20,
            trend_strength=trend_strength_regime,
            latest_bar_return=latest_bar_ret,
            volatility_breakout=vol_breakout,
            momentum_acceleration=mom_accel,
            mean_reversion_strength=mr_strength,
            bollinger_band_distance=bb_dist,
            range_expansion=range_exp,
            price_vs_recent_high_low=price_vs_range,
            ema_crossover_strength=ema_cross,
            funding_rate=fund_r,
            oi_change=oi_ch,
            symbol=feature_vector.symbol,
        )
        self._event_bus.publish("market.state", market_state_vector)

    def _rolling_volatility_baseline(self) -> float:
        if not self._rolling_realized_vol:
            return _EPS
        return max(sum(self._rolling_realized_vol) / len(self._rolling_realized_vol), _EPS)

    def _amplified_trend_state(
        self,
        trend_strength: float,
        momentum: float,
        realized_vol: float,
        rolling_vol: float,
    ) -> float:
        """Amplify weak signals: normalize, tanh, momentum confirmation, regime filter."""
        raw = self._sanitize(trend_strength)
        self._rolling_trend_raw.append(raw)

        # 1. Normalize: signal = raw / rolling_std
        std = _rolling_std(self._rolling_trend_raw)
        normalized = raw / std if std >= _EPS else 0.0

        # 2. Non-linearity: tanh(signal * k)
        signal = tanh(normalized * self.AMPLIFY_TANH_K)

        # 3. Momentum confirmation: short MA (5) vs long MA (20)
        if len(self._rolling_trend_raw) >= self.AMPLIFY_MOMENTUM_LONG:
            short_ma = _rolling_mean(self._rolling_trend_raw, self.AMPLIFY_MOMENTUM_SHORT)
            long_ma = _rolling_mean(self._rolling_trend_raw, self.AMPLIFY_MOMENTUM_LONG)
            if short_ma > long_ma:
                signal = signal * (1.0 + self.AMPLIFY_MOMENTUM_BOOST)
            elif short_ma < long_ma:
                signal = signal * (1.0 - self.AMPLIFY_MOMENTUM_BOOST)
        signal = max(-1.0, min(1.0, signal))

        # 4. Regime filter: low vol -> reduce (avoid noise), high vol -> full
        vol_ratio = realized_vol / max(rolling_vol, _EPS)
        if vol_ratio < self.AMPLIFY_VOL_THRESHOLD:
            signal *= self.AMPLIFY_LOW_VOL_REDUCE
        signal = max(-1.0, min(1.0, signal))

        # Map [-1, 1] to [0, 1] for trend_state
        return max(0.0, min(1.0, (signal + 1.0) / 2.0))

    def _volatility_state(self, realized_vol: float, rolling_vol: float) -> float:
        ratio = realized_vol / max(rolling_vol, _EPS)
        return minmax_scale(ratio, self.VOL_RATIO_MIN, self.VOL_RATIO_MAX)

    def _sanitize(self, value: float) -> float:
        return float(value) if isfinite(value) else 0.0

    def _sanitize_nonnegative(self, value: float) -> float:
        return max(0.0, self._sanitize(value))
