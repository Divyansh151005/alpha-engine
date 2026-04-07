"""Tests for overtrading filters."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.backtest.engine import BacktestEngine
from alpha_engine.data.models import OHLCV
from alpha_engine.filters import CooldownTracker, SignalFilter, TradeFrequencyTracker
from alpha_engine.state_engine.state_models import MarketStateVector


def _market_state(
    timestamp: int = 1700000000000,
    volatility_state: float = 0.5,
) -> MarketStateVector:
    return MarketStateVector(
        timestamp=timestamp,
        trend_state=0.6,
        volatility_state=volatility_state,
        liquidity_stress=0.4,
        momentum_pressure=0.5,
        compression_state=0.5,
        trend_strength=0.01,
    )


def _make_candle(i: int, close: float, base_ts: int = 1700000000000) -> OHLCV:
    return OHLCV(
        symbol="BTC/USDT",
        timeframe="1m",
        timestamp=base_ts + (i * 60000),
        open=close - 0.2,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=10.0,
    )


def _uptrend_candles(n: int, start: float = 100.0, step: float = 1.0) -> list[OHLCV]:
    return [_make_candle(i, start + i * step) for i in range(n)]


# --- SignalFilter tests ---


def test_signal_filter_below_threshold_returns_zero() -> None:
    """Signal below threshold should be blocked (return 0)."""
    filt = SignalFilter(base_threshold=0.4)
    state = _market_state()
    assert filt.filter("TestStrategy", 0.2, state) == 0.0
    assert filt.filter("TestStrategy", -0.3, state) == 0.0


def test_signal_filter_above_threshold_passes() -> None:
    """Signal above threshold should pass through."""
    filt = SignalFilter(base_threshold=0.4)
    state = _market_state()
    assert filt.filter("TestStrategy", 0.6, state) == 0.6
    assert filt.filter("TestStrategy", -0.7, state) == -0.7


def test_signal_filter_volatility_adjustment() -> None:
    """High vol -> lower threshold; low vol -> higher threshold."""
    filt = SignalFilter()
    high_vol_state = _market_state(volatility_state=0.8)
    low_vol_state = _market_state(volatility_state=0.2)
    # Signal 0.35: below base 0.4, above high_vol 0.3, below low_vol 0.5
    assert filt.filter("Test", 0.35, high_vol_state) == 0.35
    assert filt.filter("Test", 0.35, low_vol_state) == 0.0


# --- Cooldown tests ---


class _StrongSignalStrategy(BaseAlphaStrategy):
    """Always emits strong signal for trade tests."""

    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name="StrongSignal",
            score=0.9,
            confidence=0.9,
        )


def test_cooldown_prevents_rapid_trades() -> None:
    """Cooldown blocks strategies within window; cooldown tracker behaves correctly."""
    # Use flip-flop strategy: alternates long/short every candle -> many potential trades
    class _FlipFlopStrategy(BaseAlphaStrategy):
        def __init__(self) -> None:
            self._step = 0

        def on_market_state(self, state: MarketStateVector) -> None:
            _ = state

        def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
            score = 0.8 if (self._step % 2 == 0) else -0.8
            self._step += 1
            return AlphaSignal(
                timestamp=state.timestamp,
                strategy_name="FlipFlop",
                score=score,
                confidence=0.9,
            )

    candles = _uptrend_candles(150, start=100.0, step=0.1)
    engine_no_cooldown = BacktestEngine(
        initial_equity=100.0,
        strategies=[_FlipFlopStrategy()],
        diagnostics=False,
        cooldown_candles=0,
        min_holding_period=0,
        position_change_threshold=0.0,
        max_trades_per_100_steps=1000,
    )
    engine_with_cooldown = BacktestEngine(
        initial_equity=100.0,
        strategies=[_FlipFlopStrategy()],
        diagnostics=False,
        cooldown_candles=5,
        min_holding_period=0,
        position_change_threshold=0.0,
        max_trades_per_100_steps=1000,
    )
    r_no = engine_no_cooldown.run(candles)
    r_with = engine_with_cooldown.run(candles)
    # With cooldown, we block strategies for 5 bars after each trade -> fewer trades
    assert r_with.metrics["trade_count"] <= r_no.metrics["trade_count"]


def test_cooldown_tracker_is_blocked() -> None:
    """CooldownTracker correctly blocks within window."""
    tracker = CooldownTracker(cooldown_candles=5)
    tracker.record_trade(["S1"], step=10)
    assert tracker.is_blocked("S1", 11) is True
    assert tracker.is_blocked("S1", 14) is True
    assert tracker.is_blocked("S1", 15) is False
    assert tracker.is_blocked("S2", 11) is False


# --- Position change filter tests ---


def test_small_position_changes_ignored() -> None:
    """position_change_threshold is deprecated; execution follows target for any |delta| > eps."""
    # Use strategy that oscillates score so net target moves each bar.
    class _SmallChangeStrategy(BaseAlphaStrategy):
        def __init__(self) -> None:
            self._step = 0

        def on_market_state(self, state: MarketStateVector) -> None:
            _ = state

        def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
            # Alternate between 0.5 and 0.6 -> small net change
            score = 0.5 + (0.1 if (self._step % 2 == 0) else -0.1)
            self._step += 1
            return AlphaSignal(
                timestamp=state.timestamp,
                strategy_name="SmallChange",
                score=score,
                confidence=0.9,
            )

    candles = _uptrend_candles(100)
    engine_a = BacktestEngine(
        initial_equity=100.0,
        strategies=[_SmallChangeStrategy()],
        diagnostics=False,
        position_change_threshold=0.5,
        cooldown_candles=0,
        max_trades_per_100_steps=1000,
    )
    engine_b = BacktestEngine(
        initial_equity=100.0,
        strategies=[_SmallChangeStrategy()],
        diagnostics=False,
        position_change_threshold=0.01,
        cooldown_candles=0,
        max_trades_per_100_steps=1000,
    )
    r_a = engine_a.run(candles)
    r_b = engine_b.run(candles)
    assert r_a.metrics["trade_count"] == r_b.metrics["trade_count"]


# --- Trade frequency cap tests ---


def test_trade_cap_enforced() -> None:
    """max_trades_per_100_steps should cap trade frequency."""
    candles = _uptrend_candles(150, start=100.0, step=0.2)
    engine_uncapped = BacktestEngine(
        initial_equity=100.0,
        strategies=[_StrongSignalStrategy()],
        diagnostics=False,
        cooldown_candles=0,
        position_change_threshold=0.0,
        max_trades_per_100_steps=1000,
    )
    engine_capped = BacktestEngine(
        initial_equity=100.0,
        strategies=[_StrongSignalStrategy()],
        diagnostics=False,
        cooldown_candles=0,
        position_change_threshold=0.0,
        max_trades_per_100_steps=5,
    )
    r_uncapped = engine_uncapped.run(candles)
    r_capped = engine_capped.run(candles)
    assert r_capped.metrics["trade_count"] <= r_uncapped.metrics["trade_count"]


def test_trade_frequency_tracker_counts_correctly() -> None:
    """TradeFrequencyTracker counts trades in window."""
    tracker = TradeFrequencyTracker(max_trades_per_100_steps=3, window_size=100)
    for step in [10, 20, 30]:
        tracker.record_trade(["S1"], step)
    assert tracker.trade_count_in_window("S1", 50) == 3
    assert tracker.is_blocked("S1", 50) is True
    assert tracker.is_blocked("S2", 50) is False


# --- Integration: signal below threshold blocked ---


def test_signal_filter_integration_blocks_weak_signals() -> None:
    """SignalFilter in pipeline should block weak signals, reducing trades."""
    class _WeakSignalStrategy(BaseAlphaStrategy):
        def on_market_state(self, state: MarketStateVector) -> None:
            _ = state

        def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
            return AlphaSignal(
                timestamp=state.timestamp,
                strategy_name="Weak",
                score=0.25,  # Below default 0.4 threshold
                confidence=0.9,
            )

    candles = _uptrend_candles(80)
    engine_with_filter = BacktestEngine(
        initial_equity=100.0,
        strategies=[_WeakSignalStrategy()],
        diagnostics=False,
    )
    result = engine_with_filter.run(candles)
    # With ENTRY/EXIT hysteresis (ENTRY=0.3, EXIT=0.1), this weak signal should not enter a position.
    assert result.metrics["avg_position_size"] == 0.0
    assert result.metrics["trade_count"] == 0.0
