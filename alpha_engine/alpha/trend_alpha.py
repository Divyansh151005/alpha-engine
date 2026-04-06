"""Trend alpha: multi-TF momentum, EMA crossover, normalized trend. Output [-1, 1]."""

from __future__ import annotations

import math

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.state_engine.state_models import MarketStateVector


class TrendAlphaStrategy(BaseAlphaStrategy):
    """Score-based trend: multi-timeframe momentum, EMA crossover, normalized trend. Output [-1, 1]."""

    SCORE_NORMALIZE_WINDOW = 50
    _EPS = 1e-12

    def __init__(self) -> None:
        self._score_history: list[float] = []
        self._recent_scores: list[float] = []
        # Final scores after tanh/amplification/sign-flip; used for rank mapping only.
        self._rank_score_window: list[float] = []
        self._trace_emitted = 0
        self._trace_limit = 20

    def on_market_state(self, state: MarketStateVector) -> None:
        pass

    def _raw_score(self, state: MarketStateVector) -> float:
        """Combine multi-TF momentum, EMA crossover, normalized trend into raw score."""
        r5 = getattr(state, "rolling_return_5", 0.0)
        r10 = getattr(state, "rolling_return_10", 0.0)
        r20 = getattr(state, "rolling_return_20", 0.0)
        mom_scale = 20.0
        mtf_mom = (0.5 * r5 + 0.3 * r10 + 0.2 * r20) * mom_scale
        mtf_mom = max(-1.0, min(1.0, mtf_mom))

        ema_cross = getattr(state, "ema_crossover_strength", 0.0)
        ema_scaled = max(-1.0, min(1.0, ema_cross * 10.0))

        trend_norm = 2.0 * (state.trend_state - 0.5)

        raw = (mtf_mom + ema_scaled + trend_norm) / 3.0
        return max(-1.0, min(1.0, raw))

    def get_raw_signal_info(
        self, state: MarketStateVector
    ) -> tuple[str, float, float, str] | None:
        score = self._raw_score(state)
        return (
            self.__class__.__name__,
            score,
            0.0,
            "long" if score > 0 else "short" if score < 0 else "flat",
        )

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        raw = self._raw_score(state)
        self._score_history.append(raw)
        if len(self._score_history) > self.SCORE_NORMALIZE_WINDOW:
            self._score_history.pop(0)

        if len(self._score_history) >= 2:
            mean = sum(self._score_history) / len(self._score_history)
            std = max(
                math.sqrt(
                    sum((x - mean) ** 2 for x in self._score_history)
                    / (len(self._score_history) - 1)
                ),
                self._EPS,
            )
            normalized = (raw - mean) / std if std >= self._EPS else 0.0
            print(f"[TRACE] trend_normalized={normalized}")
            tanh_raw = math.tanh(normalized)
            print(f"[TRACE] tanh input={normalized} output={tanh_raw}")
            # Smooth monotonic amplification (no threshold); preserves ordering.
            score = math.copysign(abs(tanh_raw) ** 1.2, tanh_raw)
            score = max(-1.0, min(1.0, score))
        else:
            score = raw

        score = -score

        pre_rank = score
        window = self._rank_score_window + [pre_rank]
        if len(window) > 50:
            window = window[-50:]
        n = len(window)
        if n >= 2:
            sorted_scores = sorted(window)
            rank = sorted_scores.index(pre_rank)
            normalized_rank = rank / (n - 1 + 1e-6)
            score = 2.0 * normalized_rank - 1.0
            score = max(-1.0, min(1.0, score))
        else:
            score = pre_rank

        recent = list(self._recent_scores)
        recent.append(score)
        if len(recent) > 3:
            recent.pop(0)

        if len(recent) >= 2:
            last_two = recent[-2:]
            if not (
                all(s > 0 for s in last_two)
                or all(s < 0 for s in last_two)
            ):
                return None

        self._rank_score_window.append(pre_rank)
        if len(self._rank_score_window) > 50:
            self._rank_score_window.pop(0)
        self._recent_scores = recent

        if self._trace_emitted < self._trace_limit:
            step = state.timestamp
            price = state.price_vs_recent_high_low
            trend_signal = score
            print(
                "[TREND TRACE] "
                f"step={step} "
                f"price={price} "
                f"signal={trend_signal}"
            )
            self._trace_emitted += 1

        confidence = 0.3 + 0.7 * abs(score)
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name=self.__class__.__name__,
            score=score,
            confidence=confidence,
        )
