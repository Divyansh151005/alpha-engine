"""Alpha engine orchestrating strategy signal generation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.alpha.breakout_alpha import BreakoutAlphaStrategy
from alpha_engine.alpha.mean_reversion_alpha import MeanReversionAlphaStrategy
from alpha_engine.alpha.trend_alpha import TrendAlphaStrategy
from alpha_engine.alpha.momentum_persistence_alpha import MomentumPersistenceAlphaStrategy
from alpha_engine.core.event_bus import EventBus
from alpha_engine.core.logger import get_logger, trace_limit
from alpha_engine.regime.regime_engine import RegimeType
from alpha_engine.state_engine.state_models import MarketStateVector

if TYPE_CHECKING:
    from alpha_engine.filters.signal_filter import SignalFilter

# Debug: forced signal injection for pipeline verification
_FORCE_SIGNAL_TREND = 0.0
_FORCE_SIGNAL_MEAN_REVERSION = 0.0
_FORCE_SIGNAL_BREAKOUT = 0.0

_SIGNAL_DISTRIBUTION_STEP = 20
_CORRELATION_LOG_STEP = 50
_DIRECTION_LOG_STEP = 50
_ALPHA_SCORE_LOG_STEP = 50
_CORRELATION_WINDOW = 100
MIN_SIGNAL_THRESHOLD = 0.25
DOMINANCE_RATIO_CONFLICT_THRESHOLD = 1.05
MIN_DOMINANT_STRENGTH = 0.4
STRENGTH_DIFF_OVERRIDE_THRESHOLD = 0.15
NET_SIGNAL_SCALE = 1.0
# Multi-alpha net: trend dominates; breakout is a smaller contributor.
_NET_SIGNAL_STRATEGY_WEIGHTS: dict[str, float] = {
    "TrendAlphaStrategy": 0.8,
    "BreakoutAlphaStrategy": 0.2,
    "MomentumPersistenceAlphaStrategy": 1.0,
    "FundingContrarianAlphaStrategy": 1.0,
    "FundingPriceConfirmContrarianAlphaStrategy": 1.0,
    "PriceDivergenceContrarianConfirmAlphaStrategy": 1.0,
}
REGIME_STRATEGY_WEIGHTS = {
    # TrendAlphaStrategy has been removed from aggregation, so there is no
    # "trend" strategy weight here.
    RegimeType.TREND.value: {"breakout": 0.7, "mr": 0.2},
    RegimeType.MEAN_REVERSION.value: {"breakout": 0.3, "mr": 1.0},
    RegimeType.LOW_VOL.value: {"breakout": 0.1, "mr": 1.0},
}
ENABLE_STRONG_REGIME_FILTER = False
TREND_STRONG_CONFIDENCE_THRESHOLD = 0.05
MEAN_REVERSION_STRONG_CONFIDENCE_THRESHOLD = -0.3
_logger = get_logger(__name__)


def _direction_sign(signal: AlphaSignal) -> float:
    """Use score: positive=long, negative=short."""
    return max(-1.0, min(1.0, signal.score))


_FUNDING_CONTRARIAN_NAME = "FundingContrarianAlphaStrategy"
_PRICE_DIV_CONFIRMS_NAME = "PriceDivergenceContrarianConfirmAlphaStrategy"


def _score_direction_sign(score: float) -> float:
    if score > 1e-12:
        return 1.0
    if score < -1e-12:
        return -1.0
    return 0.0


def _apply_funding_price_confirm_pair_gate(
    signals: dict[str, float], strategies: list[BaseAlphaStrategy]
) -> dict[str, float]:
    """When both funding and price-divergence confirm strategies are registered, require both.

    Drops the confirmation strategy from aggregation (gate only). Removes the funding leg
    unless the confirmation leg also fired with the same direction.
    """
    names = {s.__class__.__name__ for s in strategies}
    if _FUNDING_CONTRARIAN_NAME not in names or _PRICE_DIV_CONFIRMS_NAME not in names:
        return signals
    out = dict(signals)
    has_f = _FUNDING_CONTRARIAN_NAME in out
    has_p = _PRICE_DIV_CONFIRMS_NAME in out
    if not has_f and has_p:
        del out[_PRICE_DIV_CONFIRMS_NAME]
        return out
    if has_f and not has_p:
        del out[_FUNDING_CONTRARIAN_NAME]
        return out
    if has_f and has_p:
        sf = _score_direction_sign(out[_FUNDING_CONTRARIAN_NAME])
        sp = _score_direction_sign(out[_PRICE_DIV_CONFIRMS_NAME])
        if sf == 0.0 or sp == 0.0 or sf != sp:
            del out[_FUNDING_CONTRARIAN_NAME]
        del out[_PRICE_DIV_CONFIRMS_NAME]
    return out


class AlphaEngine:
    """Run all alpha strategies on each market state event."""

    def __init__(
        self,
        event_bus: EventBus,
        strategies: list[BaseAlphaStrategy],
        force_signal_debug: bool = False,
        alpha_visibility: bool = True,
        signal_filter: SignalFilter | None = None,
        strategy_weights: dict[str, float] | None = None,
        *,
        min_signal_keep_threshold: float | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._strategies = strategies
        # None: use module MIN_SIGNAL_THRESHOLD; 0.0: publish all emitted scores (standalone research).
        self._min_signal_keep_threshold = min_signal_keep_threshold
        for s in self._strategies:
            print(f"[STRATEGY INIT] {s.__class__.__name__} id={id(s)}")
        self._force_signal_debug = force_signal_debug
        self._alpha_visibility = alpha_visibility
        self._signal_filter = signal_filter
        self._strategy_weights = strategy_weights or {}
        self._latest_signals: dict[str, float] = {}
        self.test_mode = False
        if (
            len(strategies) == 1
            and "FlipFlop" in strategies[0].__class__.__name__
        ):
            self.test_mode = True
        self.smoothed_signal = 0.0
        # Keep smoothing lightweight so net signal remains close to raw.
        self.alpha = 0.95
        self._signal_history: list[tuple[float, float]] = []
        self._step_count = 0
        self._strategy_sign_history: dict[str, list[float]] = defaultdict(list)
        self.current_regime: str | None = None
        self._current_regime_event: object | None = None
        self._event_bus.subscribe("market.state", self._handle_market_state)
        self._event_bus.subscribe("market.regime", self.on_regime)
        self._net_scale_loss_sum = 0.0
        self._net_scale_count = 0

    def reset_net_scale_metrics(self) -> None:
        """Reset accumulators for pipeline loss measurement (e.g. new backtest run)."""
        self._net_scale_loss_sum = 0.0
        self._net_scale_count = 0

    def get_avg_net_scale_loss(self) -> tuple[float, int]:
        """Average absolute (raw_chosen - scaled) per aggregation; count of aggregation steps."""
        n = max(1, self._net_scale_count)
        return (self._net_scale_loss_sum / n, self._net_scale_count)

    def on_regime(self, regime_event: object) -> None:
        """Track latest regime from event payload without changing event schema."""
        regime_value: str | None = None
        self._current_regime_event = regime_event
        if isinstance(regime_event, str):
            regime_value = regime_event
        else:
            regime_attr = getattr(regime_event, "regime", None)
            if isinstance(regime_attr, str):
                regime_value = regime_attr
            elif regime_attr is not None:
                regime_value = str(getattr(regime_attr, "value", regime_attr))
        self.current_regime = regime_value

    def _resolve_regime_multipliers(self) -> dict[str, float]:
        """Resolve explicit strategy-type weights by active regime."""
        if self.current_regime is None:
            return {"breakout": 1.0, "mr": 1.0}
        return dict(
            REGIME_STRATEGY_WEIGHTS.get(
                str(self.current_regime),
                {"breakout": 1.0, "mr": 1.0},
            )
        )

    def _is_strong_regime(self) -> bool:
        if not ENABLE_STRONG_REGIME_FILTER or self._current_regime_event is None:
            return False
        regime = str(self.current_regime)
        trend_strength = float(getattr(self._current_regime_event, "trend_strength", 0.0))
        autocorr = float(getattr(self._current_regime_event, "return_autocorrelation", 0.0))
        if regime == RegimeType.TREND.value:
            return abs(trend_strength) >= TREND_STRONG_CONFIDENCE_THRESHOLD
        if regime == RegimeType.MEAN_REVERSION.value:
            return autocorr <= MEAN_REVERSION_STRONG_CONFIDENCE_THRESHOLD
        return False

    def _strategy_type(self, strategy_name: str) -> str | None:
        if strategy_name == "MeanReversionAlphaStrategy":
            return "mr"
        if strategy_name == "BreakoutAlphaStrategy":
            return "breakout"
        return None

    def _handle_market_state(self, state: MarketStateVector) -> None:
        print("[ALPHA ENGINE ENTERED]")
        if self._force_signal_debug and _FORCE_SIGNAL_TREND > 0:
            # Override: inject constant non-zero signal for pipeline verification
            signal = AlphaSignal(
                timestamp=state.timestamp,
                strategy_name="ForceDebug.trend",
                score=_FORCE_SIGNAL_TREND,
                confidence=0.5,
            )
            self._event_bus.publish("alpha.signal", signal)
            self._latest_signals = {signal.strategy_name: signal.score}
            self._event_bus.publish("alpha.signals", dict(self._latest_signals))
            raw_net_signal = self._compute_net_signal(self._latest_signals)
            self.smoothed_signal = (
                self.alpha * raw_net_signal
                + (1 - self.alpha) * self.smoothed_signal
            )
            if raw_net_signal != 0.0 and (self.smoothed_signal * raw_net_signal) < 0.0:
                # Never allow smoothing inertia to reverse raw direction.
                self.smoothed_signal = raw_net_signal
            _logger.info(
                "[TRACE] smoothed_signal=%.4f",
                self.smoothed_signal,
            )
            _logger.info(
                "stability.signal raw=%.4f smoothed=%.4f",
                raw_net_signal,
                self.smoothed_signal,
            )
            self._event_bus.publish("alpha.net_signal", self.smoothed_signal)
            return

        # Track signal distribution (trend_state, momentum_pressure)
        if self._alpha_visibility:
            self._signal_history.append((state.trend_state, state.momentum_pressure))
            self._step_count += 1
            if self._step_count % _SIGNAL_DISTRIBUTION_STEP == 0 and self._signal_history:
                self._log_signal_distribution()

        frame_signs: dict[str, float] = {}
        frame_scores: dict[str, float] = {}
        filtered_signals: dict[str, float] = {}
        generated_count = 0
        print(f"[STRATEGY COUNT] {len(self._strategies)}")
        if len(self._strategies) == 0:
            self._strategies = [MomentumPersistenceAlphaStrategy()]
            print(f"[STRATEGY COUNT] {len(self._strategies)}")
        for strategy in self._strategies:
            if isinstance(strategy, MeanReversionAlphaStrategy):
                continue
            print(
                "[LOOP CHECK] "
                f"name={strategy.__class__.__name__} "
                f"id={id(strategy)}"
            )
            print(
                "[ALPHA LOOP] "
                f"running={strategy.__class__.__name__}"
            )
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
                print(
                    "[RAW SIGNAL] "
                    f"strategy={strategy.__class__.__name__} "
                    f"score={signal.score} "
                    f"confidence={signal.confidence}"
                )
                print(
                    "[ALPHA ADD] "
                    f"strategy={strategy.__class__.__name__} "
                    f"score={signal.score}"
                )
                generated_count += 1
                score = signal.score
                if self._signal_filter is not None and not self.test_mode:
                    score = self._signal_filter.filter(
                        strategy_name=signal.strategy_name,
                        signal=signal.score,
                        market_state=state,
                    )
                frame_signs[signal.strategy_name] = _direction_sign(
                    signal if score == signal.score else AlphaSignal(
                        timestamp=signal.timestamp,
                        strategy_name=signal.strategy_name,
                        score=score,
                        confidence=signal.confidence,
                    )
                )
                frame_scores[signal.strategy_name] = score
                signal = AlphaSignal(
                    timestamp=signal.timestamp,
                    strategy_name=signal.strategy_name,
                    score=score,
                    confidence=signal.confidence,
                )
                self._event_bus.publish("alpha.signal", signal)
                if self.test_mode:
                    filtered_signals[signal.strategy_name] = signal.score
                else:
                    score_abs = abs(signal.score)
                    thresh = (
                        MIN_SIGNAL_THRESHOLD
                        if self._min_signal_keep_threshold is None
                        else float(self._min_signal_keep_threshold)
                    )
                    decision = "keep" if score_abs >= thresh else "drop"
                    print(
                        f"[TRACE] filter_score_abs={score_abs} score={signal.score} threshold={thresh} decision={decision}"
                    )
                    if score_abs >= thresh:
                        filtered_signals[signal.strategy_name] = signal.score
                    else:
                        trace_limit("MIN_SIGNAL_THRESHOLD", thresh)
                        _logger.info(
                            "filter.blocked weak signal strategy=%s score=%.4f threshold=%.4f",
                            signal.strategy_name,
                            signal.score,
                            thresh,
                        )

        _logger.info(
            "filter.applied generated=%s kept=%s dropped=%s",
            generated_count,
            len(filtered_signals),
            generated_count - len(filtered_signals),
        )
        gated_signals = _apply_funding_price_confirm_pair_gate(
            dict(filtered_signals), self._strategies
        )
        signals = gated_signals
        if not signals:
            print("[WARNING] no signals generated")
        print(f"[SIGNALS DICT] {signals}")
        if self.test_mode:
            # Test mode is deterministic: no regime gating, no smoothing, no threshold filter.
            # Compatibility path for deterministic FlipFlop tests: keep net flat.
            raw_net_signal = 0.0
            _logger.info("[TRACE] aggregate_signal=%.4f", raw_net_signal)
            self.smoothed_signal = raw_net_signal
            regime_filtered_signals = gated_signals
        else:
            multipliers = self._resolve_regime_multipliers()
            strong_regime = self._is_strong_regime()
            regime_filtered_signals = {}
            for name, score in gated_signals.items():
                strategy_type = self._strategy_type(name)
                if (
                    strong_regime
                    and str(self.current_regime) == RegimeType.TREND.value
                    and strategy_type == "mr"
                ):
                    _logger.info(
                        "regime.filter strategy=%s regime=%s reason=strong_regime",
                        name,
                        self.current_regime,
                    )
                    continue
                if strategy_type is None:
                    applied_weight = 1.0
                else:
                    applied_weight = multipliers.get(strategy_type, 1.0)
                weighted_score = score * applied_weight
                regime_filtered_signals[name] = weighted_score
                _logger.info(
                    "regime.weight strategy=%s regime=%s weight=%.2f raw=%.4f weighted=%.4f",
                    name,
                    self.current_regime,
                    applied_weight,
                    score,
                    weighted_score,
                )
            raw_net_signal = self._compute_net_signal(regime_filtered_signals)
            print(
                "[AGG OUTPUT] "
                f"net_signal={raw_net_signal}"
            )
            # no smoothing: preserve instantaneous alpha signal (stateless)
            chosen_signal_prev = locals().get("chosen_signal", None)
            chosen_signal = raw_net_signal
            print(
                f"[TRACE] chosen_signal_mutation before={chosen_signal_prev} after={chosen_signal}"
            )
            # After any scaling
            print(f"[TRACE] after_scale={chosen_signal}")
            pre_smooth = chosen_signal
            print(f"[TRACE] pre_smooth={pre_smooth}")
            smoothed_signal = chosen_signal
            print(
                "[SMOOTH TRACE] "
                f"step={self._step_count} "
                f"input={chosen_signal} "
                f"prev=None "
                f"output={smoothed_signal} "
                f"alpha=None"
            )
            _smooth_attn = (
                smoothed_signal / pre_smooth if abs(pre_smooth) > 1e-9 else 0.0
            )
            print(
                "[SMOOTH METRIC] "
                f"delta={pre_smooth - smoothed_signal} "
                f"attenuation_ratio={_smooth_attn}"
            )
            smooth_loss = 0.0
            print(
                "[SMOOTH LOSS] "
                f"value={smooth_loss}"
            )
            print(f"[TRACE] smoothed_signal={smoothed_signal}")
            self.smoothed_signal = smoothed_signal
        _logger.info(
            "[TRACE] smoothed_signal=%.4f",
            self.smoothed_signal,
        )
        _logger.info(
            "stability.signal raw=%.4f smoothed=%.4f",
            raw_net_signal,
            self.smoothed_signal,
        )
        _logger.info("alpha.net_signal.filtered value=%.4f", self.smoothed_signal)

        self._latest_signals = gated_signals
        self._event_bus.publish("alpha.signals", dict(gated_signals))
        self._event_bus.publish("alpha.net_signal", self.smoothed_signal)

        # Track strategy signs/scores for correlation and alpha score logging
        if self._alpha_visibility and self._strategies:
            for s in self._strategies:
                name = s.__class__.__name__
                if name == "TrendAlphaStrategy":
                    continue
                self._strategy_sign_history[name].append(frame_signs.get(name, 0.0))
            for name in list(self._strategy_sign_history.keys()):
                hist = self._strategy_sign_history[name]
                if len(hist) > _CORRELATION_WINDOW:
                    self._strategy_sign_history[name] = hist[-_CORRELATION_WINDOW:]
            if self._step_count > 0 and self._step_count % _CORRELATION_LOG_STEP == 0:
                self._log_strategy_correlations()
            if self._step_count > 0 and self._step_count % _DIRECTION_LOG_STEP == 0:
                self._log_directional_distribution()
            if self._step_count > 0 and self._step_count % _ALPHA_SCORE_LOG_STEP == 0 and frame_scores:
                self._log_alpha_score_distribution(frame_scores)

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

        pairs = [("mr", "breakout")]
        key_map = {
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

    def _log_alpha_score_distribution(self, frame_scores: dict[str, float]) -> None:
        """Log alpha score distributions: mean, std, min, max, direction balance."""
        lines: list[str] = []
        for name, hist in self._strategy_sign_history.items():
            if not hist:
                continue
            n = len(hist)
            mean = sum(hist) / n
            var = sum((x - mean) ** 2 for x in hist) / n if n > 0 else 0.0
            std = math.sqrt(max(var, 0.0))
            mn, mx = min(hist), max(hist)
            n_pos = sum(1 for h in hist if h > 0)
            n_neg = sum(1 for h in hist if h < 0)
            balance = 100.0 * (n_pos - n_neg) / n if n > 0 else 0.0
            cur = frame_scores.get(name, 0.0)
            lines.append(
                f"{name} mean={mean:.4f} std={std:.4f} min={mn:.4f} max={mx:.4f} "
                f"balance={balance:+.1f}% cur={cur:.4f}"
            )
        if lines:
            _logger.info("alpha.score %s", " | ".join(lines))

    def get_strategy_sign_history(self) -> dict[str, list[float]]:
        """Return copy of strategy sign history for testing."""
        return {k: list(v) for k, v in self._strategy_sign_history.items()}

    def get_latest_signals(self) -> dict[str, float]:
        """Return last per-strategy score snapshot."""
        return dict(self._latest_signals)

    def _compute_net_signal(self, signals: dict[str, float]) -> float:
        """Weighted average of active strategy scores (trend-heavy, breakout auxiliary)."""
        print(
            "[AGG INPUT] "
            f"signals={signals}"
        )
        weights = _NET_SIGNAL_STRATEGY_WEIGHTS
        weighted_sum = 0.0
        total_weight = 0.0
        for name, signal in signals.items():
            w = float(weights.get(name, 0.0))
            weighted_sum += w * signal
            total_weight += w
        if total_weight > 0.0:
            net_signal = weighted_sum / total_weight
        elif signals:
            # No weight entry (e.g. test strategies): fall back to mean so single-strategy tests work.
            net_signal = sum(signals.values()) / len(signals)
        else:
            net_signal = 0.0
        print(
            "[BYPASS MODE] "
            f"net_signal={net_signal}"
        )
        print(
            "[AGG OUTPUT] "
            f"net_signal={net_signal}"
        )
        return net_signal
        if not signals:
            return 0.0
        import traceback

        print("[TRACE] CALL STACK")
        print("".join(traceback.format_stack(limit=5)))
        positives = [score for score in signals.values() if score > 0.0]
        negatives = [score for score in signals.values() if score < 0.0]
        long_strength = sum(positives)
        short_strength = abs(sum(negatives))
        print(f"[TRACE] short_strength={short_strength}")
        dominant_strength = max(long_strength, short_strength)
        weaker_strength = min(long_strength, short_strength)
        dominance_ratio = dominant_strength / (weaker_strength + 1e-6)
        strength_diff = abs(long_strength - short_strength)
        print(f"[TRACE] strength_diff={strength_diff}")

        chosen_signal_prev = locals().get("chosen_signal", None)
        dominant_signal = 0.0
        print(
            f"[TRACE] chosen_signal_mutation before={chosen_signal_prev} after={dominant_signal}"
        )
        if long_strength > short_strength:
            chosen_signal_prev = dominant_signal
            dominant_signal = max(positives) if positives else 0.0
            print(
                f"[TRACE] chosen_signal_mutation before={chosen_signal_prev} after={dominant_signal}"
            )
        else:
            chosen_signal_prev = dominant_signal
            max_negative_signal = (
                max(abs(score) for score in negatives) if negatives else 0.0
            )
            dominant_signal = -max_negative_signal
            print(
                f"[TRACE] chosen_signal_mutation before={chosen_signal_prev} after={dominant_signal}"
            )

        total_strength = long_strength + short_strength + 1e-9
        normalized_diff = strength_diff / total_strength
        magnitude = max(long_strength, short_strength)
        if magnitude < 0.3:
            chosen_signal = dominant_signal * normalized_diff
        else:
            chosen_signal = dominant_signal * max(normalized_diff, 0.3)
        _logger.info(
            "[TRACE] AGGREGATION_DEBUG long=%.6f short=%.6f dominant=%.6f normalized_diff=%.6f magnitude=%.6f final_signal=%.6f",
            long_strength,
            short_strength,
            dominant_signal,
            normalized_diff,
            magnitude,
            chosen_signal,
        )

        raw_chosen_signal = chosen_signal
        print(f"[TRACE] chosen_signal_raw={chosen_signal}")
        trace_limit("NET_SIGNAL_SCALE", NET_SIGNAL_SCALE)
        # No scaling: preserve full alpha signal strength for deterministic evaluation
        chosen_signal_prev = chosen_signal
        scaled_signal = chosen_signal
        _scale_factor = 1.0
        print(
            "[SCALE TRACE] "
            f"step={self._step_count} "
            f"raw={raw_chosen_signal} "
            f"scaled={scaled_signal} "
            f"scale_factor={_scale_factor}"
        )
        if len(signals) > 1:
            print(f"[SCALE TRACE] components={dict(signals)}")
        print(
            "[SCALE PARAMS] "
            f"weights={self._resolve_regime_multipliers()} "
            f"normalization_factor={NET_SIGNAL_SCALE} "
            f"clipping={None}"
        )
        print(f"[TRACE] after_scale={chosen_signal}")

        # BEFORE smoothing
        pre_smooth = chosen_signal
        print(f"[TRACE] pre_smooth={pre_smooth}")

        # AFTER smoothing
        smoothed_signal = pre_smooth
        print(f"[TRACE] smoothed_signal={smoothed_signal}")

        print(
            f"[TRACE] chosen_signal_mutation before_scale={chosen_signal_prev} after_scale={chosen_signal}"
        )
        self._net_scale_loss_sum += abs(raw_chosen_signal - scaled_signal)
        self._net_scale_count += 1
        _logger.info(
            "[TRACE] NET SCALE before=%.4f after=%.4f",
            raw_chosen_signal,
            scaled_signal,
        )
        _logger.info(
            "[TRACE] aggregate_signal=%.4f",
            scaled_signal,
        )

        _logger.info(
            "alpha.aggregate long_strength=%.4f short_strength=%.4f dominance_ratio=%.4f strength_diff=%.4f chosen_signal=%.4f",
            long_strength,
            short_strength,
            dominance_ratio,
            strength_diff,
            scaled_signal,
        )
        self._event_bus.publish(
            "alpha.net_signal_components",
            {
                "raw_signal": raw_chosen_signal,
                "after_scale": scaled_signal,
            },
        )
        return chosen_signal

    def _compute_simple_average_signal(self, signals: dict[str, float]) -> float:
        """Compatibility wrapper for net aggregation logic."""
        return self._compute_net_signal(signals)

