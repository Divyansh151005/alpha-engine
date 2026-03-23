"""Alpha engine orchestrating strategy signal generation."""

from __future__ import annotations

import math
from collections import defaultdict

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.logger import get_logger
from alpha_engine.state_engine.state_models import MarketStateVector

# Debug: forced signal injection for pipeline verification
_FORCE_SIGNAL_TREND = 0.5
_FORCE_SIGNAL_MEAN_REVERSION = 0.0
_FORCE_SIGNAL_BREAKOUT = 0.0

_SIGNAL_DISTRIBUTION_STEP = 20
_CORRELATION_LOG_STEP = 50
_DIRECTION_LOG_STEP = 50
_CORRELATION_WINDOW = 100
_logger = get_logger(__name__)


def _direction_sign(direction: str) -> float:
    if direction == "long":
        return 1.0
    if direction == "short":
        return -1.0
    return 0.0


class AlphaEngine:
    """Run all alpha strategies on each market state event."""

    def __init__(
        self,
        event_bus: EventBus,
        strategies: list[BaseAlphaStrategy],
        force_signal_debug: bool = False,
        alpha_visibility: bool = True,
    ) -> None:
        self._event_bus = event_bus
        self._strategies = strategies
        self._force_signal_debug = force_signal_debug
        self._alpha_visibility = alpha_visibility
        self._signal_history: list[tuple[float, float]] = []
        self._step_count = 0
        self._strategy_sign_history: dict[str, list[float]] = defaultdict(list)
        self._event_bus.subscribe("market.state", self._handle_market_state)

    def _handle_market_state(self, state: MarketStateVector) -> None:
        if self._force_signal_debug:
            # Override: inject constant non-zero signal for pipeline verification
            if _FORCE_SIGNAL_TREND > 0:
                signal = AlphaSignal(
                    timestamp=state.timestamp,
                    strategy_name="ForceDebug.trend",
                    direction="long",
                    strength=_FORCE_SIGNAL_TREND,
                    confidence=0.5,
                )
                self._event_bus.publish("alpha.signal", signal)
            return

        # Track signal distribution (trend_state, momentum_pressure)
        if self._alpha_visibility:
            self._signal_history.append((state.trend_state, state.momentum_pressure))
            self._step_count += 1
            if self._step_count % _SIGNAL_DISTRIBUTION_STEP == 0 and self._signal_history:
                self._log_signal_distribution()

        frame_signs: dict[str, float] = {}
        for strategy in self._strategies:
            strategy.on_market_state(state)
            signal = strategy.generate_signal(state)
            if self._alpha_visibility and hasattr(strategy, "get_raw_signal_info"):
                raw_info = strategy.get_raw_signal_info(state)
                if raw_info:
                    name, value, threshold, decision = raw_info
                    _logger.info(
                        "alpha.raw strategy=%s value=%.6f threshold=%.6f decision=%s",
                        name,
                        value,
                        threshold,
                        decision,
                    )
            if signal is not None:
                frame_signs[signal.strategy_name] = _direction_sign(signal.direction)
                self._event_bus.publish("alpha.signal", signal)

        # Track strategy signs for correlation (one value per bar per strategy)
        if self._alpha_visibility and self._strategies:
            for s in self._strategies:
                name = s.__class__.__name__
                self._strategy_sign_history[name].append(frame_signs.get(name, 0.0))
            for name in list(self._strategy_sign_history.keys()):
                hist = self._strategy_sign_history[name]
                if len(hist) > _CORRELATION_WINDOW:
                    self._strategy_sign_history[name] = hist[-_CORRELATION_WINDOW:]
            if self._step_count > 0 and self._step_count % _CORRELATION_LOG_STEP == 0:
                self._log_strategy_correlations()
            if self._step_count > 0 and self._step_count % _DIRECTION_LOG_STEP == 0:
                self._log_directional_distribution()

    def _log_signal_distribution(self) -> None:
        """Log mean, std, min, max of trend_state and momentum_pressure."""
        if not self._signal_history:
            return
        trend_vals = [s[0] for s in self._signal_history]
        mom_vals = [s[1] for s in self._signal_history]

        def _stats(vals: list[float]) -> tuple[float, float, float, float]:
            n = len(vals)
            if n == 0:
                return 0.0, 0.0, 0.0, 0.0
            mean = sum(vals) / n
            var = sum((x - mean) ** 2 for x in vals) / n if n > 0 else 0.0
            std = math.sqrt(max(var, 0.0))
            return mean, std, min(vals), max(vals)

        t_mean, t_std, t_min, t_max = _stats(trend_vals)
        m_mean, m_std, m_min, m_max = _stats(mom_vals)
        _logger.info(
            "alpha.distribution step=%s trend mean=%.4f std=%.4f min=%.4f max=%.4f | momentum mean=%.4f std=%.4f min=%.4f max=%.4f",
            self._step_count,
            t_mean,
            t_std,
            t_min,
            t_max,
            m_mean,
            m_std,
            m_min,
            m_max,
        )

    def _log_strategy_correlations(self) -> None:
        """Log correlation between strategy signal series."""
        names = list(self._strategy_sign_history.keys())
        if len(names) < 2:
            return
        n = min(len(self._strategy_sign_history[n]) for n in names)
        if n < 10:
            return
        series = {k: v[-n:] for k, v in self._strategy_sign_history.items()}

        def _corr(a: list[float], b: list[float]) -> float:
            ma = sum(a) / len(a)
            mb = sum(b) / len(b)
            cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
            va = sum((x - ma) ** 2 for x in a) / len(a)
            vb = sum((y - mb) ** 2 for y in b) / len(b)
            denom = math.sqrt(va * vb)
            return cov / denom if denom > 1e-12 else 0.0

        pairs = [
            ("trend", "mr"),
            ("trend", "breakout"),
            ("mr", "breakout"),
        ]
        key_map = {
            "trend": "TrendAlphaStrategy",
            "mr": "MeanReversionAlphaStrategy",
            "breakout": "BreakoutAlphaStrategy",
        }
        corrs: list[str] = []
        for a, b in pairs:
            ka, kb = key_map.get(a, a), key_map.get(b, b)
            if ka in series and kb in series:
                c = _corr(series[ka], series[kb])
                corrs.append(f"corr({a},{b})={c:.3f}")
        if corrs:
            _logger.info("alpha.correlation %s", " ".join(corrs))

    def _log_directional_distribution(self) -> None:
        """Log % long and % short signals per strategy."""
        lines: list[str] = []
        for name, hist in self._strategy_sign_history.items():
            recent = [h for h in hist if h != 0.0]
            if not recent:
                lines.append(f"{name}=n/a")
                continue
            n_long = sum(1 for h in recent if h > 0)
            n_short = sum(1 for h in recent if h < 0)
            pct_long = 100.0 * n_long / len(recent)
            pct_short = 100.0 * n_short / len(recent)
            lines.append(f"{name} long={pct_long:.1f}% short={pct_short:.1f}%")
        if lines:
            _logger.info("alpha.direction %s", " | ".join(lines))

    def get_strategy_sign_history(self) -> dict[str, list[float]]:
        """Return copy of strategy sign history for testing."""
        return {k: list(v) for k, v in self._strategy_sign_history.items()}


class TrendAlphaStrategy(BaseAlphaStrategy):
    """Trend-following strategy over amplified signals with dynamic threshold."""

    MOMENTUM_NEUTRAL = 0.5  # momentum_pressure midpoint: above = bullish, below = bearish
    SIGNAL_WINDOW = 50
    DYNAMIC_THRESHOLD_FACTOR = 0.5
    _EPS = 1e-12

    def __init__(self) -> None:
        self._signal_history: list[float] = []

    def on_market_state(self, state: MarketStateVector) -> None:
        self._signal_history.append(state.trend_state)
        if len(self._signal_history) > self.SIGNAL_WINDOW:
            self._signal_history.pop(0)

    def _dynamic_threshold(self) -> float:
        """threshold = 0.5 * std(signal)."""
        if len(self._signal_history) < 2:
            return 0.1
        n = len(self._signal_history)
        mean = sum(self._signal_history) / n
        var = sum((x - mean) ** 2 for x in self._signal_history) / (n - 1)
        std = max(math.sqrt(max(var, 0.0)), self._EPS)
        return self.DYNAMIC_THRESHOLD_FACTOR * std

    def get_raw_signal_info(
        self, state: MarketStateVector
    ) -> tuple[str, float, float, str] | None:
        """Return (name, primary_value, threshold, decision) for alpha.raw logging."""
        value = state.trend_state
        mom = state.momentum_pressure
        thresh = self._dynamic_threshold()
        if value > (0.5 + thresh) and mom > self.MOMENTUM_NEUTRAL:
            return (self.__class__.__name__, value, 0.5 + thresh, "long")
        if value < (0.5 - thresh) and mom < self.MOMENTUM_NEUTRAL:
            return (self.__class__.__name__, value, 0.5 - thresh, "short")
        return (self.__class__.__name__, value, thresh, "none")

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        thresh = self._dynamic_threshold()
        # Long: uptrend + momentum confirming up. Short: downtrend + momentum confirming down.
        if (
            state.trend_state > 0.5 + thresh
            and state.momentum_pressure > self.MOMENTUM_NEUTRAL
        ):
            direction = "long"
        elif (
            state.trend_state < 0.5 - thresh
            and state.momentum_pressure < self.MOMENTUM_NEUTRAL
        ):
            direction = "short"
        else:
            return None

        # Strength/confidence: positive for long, symmetric for short (avoid bias)
        if direction == "long":
            strength = state.trend_state
            confidence = state.momentum_pressure
        else:
            strength = 1.0 - state.trend_state
            confidence = 1.0 - state.momentum_pressure

        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self.__class__.__name__,
            direction=direction,
            strength=max(0.01, min(1.0, strength)),
            confidence=max(0.01, min(1.0, confidence)),
        )
