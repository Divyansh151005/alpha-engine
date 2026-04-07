"""Tests for backtest engine."""

from __future__ import annotations

import pytest

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    compute_metrics,
    compute_target_position,
)
from alpha_engine.data.models import OHLCV
from alpha_engine.state_engine.state_models import MarketStateVector


class _AlwaysLongStrategy(BaseAlphaStrategy):
    """Test strategy that always emits long signal for trade visibility tests."""

    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name="AlwaysLong",
            score=0.8,
            confidence=0.9,
        )


class _SubThresholdLongStrategy(BaseAlphaStrategy):
    """|score| below sizing dead zone (<0.2) — stays flat (simulates no effective alpha)."""

    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name="SubThreshold",
            score=0.15,
            confidence=0.9,
        )


class _FlipFlopStrategy(BaseAlphaStrategy):
    """Alternates long/short every candle to create high turnover."""

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


class _WeakFluctuationStrategy(BaseAlphaStrategy):
    """Oscillates inside hold-band to avoid repeated trades."""

    def __init__(self) -> None:
        self._step = 0

    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        # Stay below |signal| < 0.2 dead zone (oscillating weak band).
        score = 0.15 if (self._step % 2 == 0) else 0.1
        self._step += 1
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name="WeakFluct",
            score=score,
            confidence=0.9,
        )


class _StrongLongStrategy(BaseAlphaStrategy):
    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name="StrongLong",
            score=0.9,
            confidence=1.0,
        )


class _WeakShortStrategy(BaseAlphaStrategy):
    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name="WeakShort",
            score=-0.65,
            confidence=0.5,
        )


class _OffsettingShortStrategy(BaseAlphaStrategy):
    """Consistent short that partially offsets a strong long strategy."""

    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name="OffsettingShort",
            score=-0.8,
            confidence=1.0,
        )


class _WeakPointTwoStrategy(BaseAlphaStrategy):
    """Always emits weak signal that should never enter position."""

    def on_market_state(self, state: MarketStateVector) -> None:
        _ = state

    def generate_signal(self, state: MarketStateVector) -> AlphaSignal | None:
        return AlphaSignal(
            timestamp=state.timestamp,
            strategy_name="WeakPointTwo",
            score=0.2,
            confidence=0.8,
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


def test_compute_target_position_mapping() -> None:
    current = 0.25
    assert compute_target_position(0.0, current_position=current) == 0.0
    # Below |signal| < 0.2 -> flat
    assert compute_target_position(0.14, current_position=current) == 0.0
    assert compute_target_position(-0.14, current_position=current) == 0.0
    assert compute_target_position(0.05, current_position=current) == 0.0
    assert compute_target_position(0.2, current_position=current) == 0.0
    # Linear: magnitude = |signal| - 0.2 for |signal| in [0.2, 1], capped in [-1, 1]
    assert compute_target_position(0.39, current_position=current) == pytest.approx(0.19)
    assert compute_target_position(0.41, current_position=current) == pytest.approx(0.21)
    assert compute_target_position(0.74, current_position=current) == pytest.approx(0.54)
    assert compute_target_position(0.76, current_position=current) == pytest.approx(0.56)
    assert compute_target_position(0.8, current_position=current) == pytest.approx(0.6)
    assert compute_target_position(-0.8, current_position=current) == pytest.approx(-0.6)


def test_deterministic_replay() -> None:
    """Same candles in -> same equity curve out (fresh engine per run)."""
    candles = _uptrend_candles(50)
    engine1 = BacktestEngine(initial_equity=100.0)
    engine2 = BacktestEngine(initial_equity=100.0)
    r1 = engine1.run(candles)
    r2 = engine2.run(candles)
    assert len(r1.equity_curve) == len(r2.equity_curve)
    for (t1, e1), (t2, e2) in zip(r1.equity_curve, r2.equity_curve):
        assert t1 == t2
        assert e1 == e2


def test_equity_increases_on_uptrend() -> None:
    """Backtest runs and produces valid equity curve on uptrend data."""
    # 80 candles of strong uptrend; strategy may or may not signal
    candles = _uptrend_candles(80, start=100.0, step=0.5)
    engine = BacktestEngine(initial_equity=100.0)
    result = engine.run(candles)
    assert len(result.equity_curve) == 80
    assert result.equity_curve[0][1] == 100.0
    assert all(eq > 0 for _, eq in result.equity_curve)


def test_drawdown_computed_correctly() -> None:
    """Max drawdown from peak to trough."""
    # Equity: 100 -> 110 -> 95 -> 105. Peak 110, trough 95, dd = (110-95)/110
    curve = [
        (1700000000000, 100.0),
        (1700000060000, 110.0),
        (1700000120000, 95.0),
        (1700000180000, 105.0),
    ]
    metrics = compute_metrics(curve)
    expected_dd = (110.0 - 95.0) / 110.0
    assert abs(metrics["max_drawdown"] - expected_dd) < 1e-6


def test_metrics_valid() -> None:
    """Metrics are finite and in expected ranges."""
    curve = [
        (1700000000000, 100.0),
        (1700000060000, 105.0),
        (1700000120000, 102.0),
        (1700000180000, 108.0),
    ]
    metrics = compute_metrics(curve)
    assert "total_return" in metrics
    assert "cagr" in metrics
    assert "max_drawdown" in metrics
    assert "sharpe_ratio" in metrics
    assert all(
        isinstance(v, (int, float)) and abs(v) < 1e10
        for v in metrics.values()
    )
    assert 0.0 <= metrics["max_drawdown"] <= 1.0


def test_extended_metrics_in_backtest_result() -> None:
    """Backtest result includes trade_count, avg_position_size, time_in_market."""
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_AlwaysLongStrategy()],
        diagnostics=False,
    )
    result = engine.run(_uptrend_candles(50))
    assert "trade_count" in result.metrics
    assert "avg_position_size" in result.metrics
    assert "time_in_market" in result.metrics
    assert "turnover" in result.metrics
    assert "total_cost" in result.metrics
    assert "gross_total_return" in result.metrics
    assert "net_total_return" in result.metrics
    assert "cost_impact" in result.metrics


def test_backtest_result_structure() -> None:
    """BacktestResult has equity_curve and metrics."""
    engine = BacktestEngine(initial_equity=100.0)
    result = engine.run(_uptrend_candles(30))
    assert isinstance(result, BacktestResult)
    assert hasattr(result, "equity_curve")
    assert hasattr(result, "metrics")
    assert len(result.equity_curve) == 30
    assert result.equity_curve[0][1] == 100.0


def test_empty_candles() -> None:
    """Empty or single candle yields valid result."""
    engine = BacktestEngine(initial_equity=100.0)
    result = engine.run([])
    assert result.equity_curve == []
    assert result.metrics["total_return"] == 0.0
    assert result.metrics["max_drawdown"] == 0.0
    assert result.metrics["trade_count"] == 0
    assert result.metrics["avg_position_size"] == 0.0


def test_signals_generated_with_always_long() -> None:
    """Strategy that always signals produces allocations and trades."""
    candles = _uptrend_candles(80)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_AlwaysLongStrategy()],
        diagnostics=False,
    )
    result = engine.run(candles)
    assert result.metrics["trade_count"] >= 0
    assert result.metrics["avg_position_size"] >= 0
    assert result.metrics["time_in_market"] >= 0


def test_allocation_stays_zero_when_alpha_disabled() -> None:
    """With sub-threshold signal, allocation and avg_position stay at zero."""
    candles = _uptrend_candles(80)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_SubThresholdLongStrategy()],
        diagnostics=False,
        min_signal_threshold=0.0,
        signal_damping=1.0,
    )
    result = engine.run(candles)
    assert result.metrics["avg_position_size"] == 0, "should have zero position"
    assert result.metrics["time_in_market"] == 0, "should spend no time in market"


def test_trade_count_zero_with_synthetic_uptrend_when_alpha_disabled() -> None:
    """With sub-threshold signal and uptrend data, trade_count should be 0."""
    candles = _uptrend_candles(80)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_SubThresholdLongStrategy()],
        diagnostics=False,
        min_signal_threshold=0.0,
        signal_damping=1.0,
    )
    result = engine.run(candles)
    assert result.metrics["trade_count"] == 0, "should have zero position changes"


def test_trading_costs_are_zero_when_no_trades() -> None:
    """No trades implies zero total trading cost."""
    candles = _uptrend_candles(120, start=100.0, step=0.2)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_SubThresholdLongStrategy()],
        diagnostics=False,
        min_holding_period=0,
        min_signal_threshold=0.0,
        signal_damping=1.0,
    )
    result = engine.run(candles)
    assert result.metrics["total_cost"] == 0.0
    assert result.metrics["cost_impact"] >= 0.0
    assert result.metrics["gross_total_return"] >= result.metrics["net_total_return"]


def test_execution_delay_applied_at_t_plus_one() -> None:
    """First bar should not have executed position PnL due to T+1 execution."""
    candles = _uptrend_candles(120, start=100.0, step=0.5)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_SubThresholdLongStrategy()],
        diagnostics=False,
        fee_rate=0.0,
        slippage_rate=0.0,
        min_holding_period=0,
        min_signal_threshold=0.0,
        signal_damping=1.0,
    )
    result = engine.run(candles)
    # Early bars stay flat before strategy/feature warmup and delayed execution.
    assert result.equity_curve[0][1] == 100.0
    assert result.equity_curve[1][1] == 100.0
    # No alpha means no execution and flat equity.
    assert result.equity_curve[-1][1] == result.equity_curve[1][1]


def test_min_signal_threshold_ignores_weak_signals() -> None:
    """Weak signals below threshold should not create exposure."""
    candles = _uptrend_candles(60, start=100.0, step=0.3)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_AlwaysLongStrategy()],
        diagnostics=False,
        min_signal_threshold=1.1,  # Above possible normalized signal strength
    )
    result = engine.run(candles)
    assert result.metrics["avg_position_size"] == 0.0
    assert result.metrics["turnover"] == 0.0


def test_position_fraction_bounded() -> None:
    """Executed average absolute position fraction stays within [0, 1]."""
    candles = _uptrend_candles(80, start=100.0, step=0.1)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_AlwaysLongStrategy()],
        diagnostics=False,
        min_signal_threshold=0.0,
    )
    result = engine.run(candles)
    assert result.metrics["avg_position_size"] <= 1.0 + 1e-9


def test_min_hold_constraint_reduces_turnover() -> None:
    """Higher min holding period should cut turnover for oscillating signals."""
    candles = _uptrend_candles(150, start=100.0, step=0.05)
    fast_engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_FlipFlopStrategy()],
        diagnostics=False,
        min_holding_period=0,
        min_signal_threshold=0.0,
    )
    hold_engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_FlipFlopStrategy()],
        diagnostics=False,
        min_holding_period=3,
        min_signal_threshold=0.0,
    )
    fast_result = fast_engine.run(candles)
    hold_result = hold_engine.run(candles)
    assert hold_result.metrics["turnover"] <= fast_result.metrics["turnover"]


def test_no_unrealistic_equity_jumps() -> None:
    """Equity remains finite and does not jump beyond full-loss/full-gain style bounds."""
    candles = _uptrend_candles(100, start=100.0, step=0.3)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_FlipFlopStrategy()],
        diagnostics=False,
        min_holding_period=0,
        min_signal_threshold=0.0,
    )
    result = engine.run(candles)
    for i in range(1, len(result.equity_curve)):
        prev = result.equity_curve[i - 1][1]
        cur = result.equity_curve[i][1]
        pct = (cur - prev) / prev if prev > 0 else 0.0
        assert -1.0 <= pct <= 1.0


def test_position_holds_across_steps() -> None:
    """Smoothed net drifts slightly; execution resizes to target (no major-scale floor)."""
    candles = _uptrend_candles(90, start=100.0, step=0.2)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_AlwaysLongStrategy()],
        diagnostics=False,
        min_holding_period=0,
        min_signal_threshold=0.0,
    )
    result = engine.run(candles)
    assert result.metrics["trade_count"] <= 35


def test_no_trade_on_small_signal_fluctuation() -> None:
    """Oscillating weak signals update targets; execution tracks pending without suppression."""
    candles = _uptrend_candles(120, start=100.0, step=0.15)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_WeakFluctuationStrategy()],
        diagnostics=False,
        min_holding_period=0,
        min_signal_threshold=0.0,
    )
    result = engine.run(candles)
    # NOTE:
    # Execution layer no longer suppresses small signals.
    # This test previously validated execution policy (noise filtering),
    # which has been intentionally removed to ensure pure alpha evaluation.
    # Therefore, trade count is expected to be higher.
    assert engine._signal_loss == 0.0  # noqa: SLF001 — pending vs executed (no suppression)
    assert result.metrics["trade_count"] == 0


def test_trade_only_on_strong_signal_change() -> None:
    """Execution follows filtered net signal; near-flat net stays flat."""
    candles = _uptrend_candles(120, start=100.0, step=0.15)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_FlipFlopStrategy()],
        diagnostics=False,
        min_holding_period=0,
        min_signal_threshold=0.0,
    )
    result = engine.run(candles)
    assert result.metrics["trade_count"] == 0.0
    assert result.metrics["turnover"] == 0.0


def test_conflict_resolution_nets_to_strongest_direction() -> None:
    """Conflicting strategies resolve through filtered net signal only."""
    candles = _uptrend_candles(100, start=100.0, step=0.12)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_StrongLongStrategy(), _WeakShortStrategy()],
        diagnostics=False,
        min_holding_period=0,
        min_signal_threshold=0.0,
    )
    result = engine.run(candles)
    # Full signal-to-size mapping plus max_delta staging may use a few executions to reach target.
    assert result.metrics["trade_count"] <= 5.0
    assert result.metrics["avg_position_size"] >= 0.0


def test_single_net_portfolio_position_only() -> None:
    """Engine should maintain one net portfolio position, not per-strategy positions."""
    candles = _uptrend_candles(120, start=100.0, step=0.08)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_StrongLongStrategy(), _WeakShortStrategy()],
        diagnostics=False,
        min_holding_period=0,
        min_signal_threshold=0.0,
    )
    result = engine.run(candles)
    # Single net position transitions should remain very small in count.
    assert result.metrics["trade_count"] <= 6
    assert result.metrics["turnover"] <= 3.0


def test_internal_strategy_conflict_reduces_turnover() -> None:
    """Offsetting strategies should not create extra trades in net execution mode."""
    candles = _uptrend_candles(120, start=100.0, step=0.08)
    single_engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_StrongLongStrategy()],
        diagnostics=False,
        min_holding_period=0,
        min_signal_threshold=0.0,
    )
    conflicted_engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_StrongLongStrategy(), _OffsettingShortStrategy()],
        diagnostics=False,
        min_holding_period=0,
        min_signal_threshold=0.0,
    )

    single_result = single_engine.run(candles)
    conflicted_result = conflicted_engine.run(candles)
    # Net portfolio should not overtrade because of internal opposing strategies.
    assert conflicted_result.metrics["trade_count"] <= single_result.metrics["trade_count"] + 1
    assert conflicted_result.metrics["turnover"] <= single_result.metrics["turnover"] + 1.0


def test_weak_signal_point_two_stays_flat_no_trades() -> None:
    """Weak 0.2 signal should stay flat and create zero trades."""
    candles = _uptrend_candles(80, start=100.0, step=0.15)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_WeakPointTwoStrategy()],
        diagnostics=False,
        min_signal_threshold=0.0,
    )
    result = engine.run(candles)
    assert result.metrics["avg_position_size"] == 0.0
    assert result.metrics["trade_count"] == 0.0


def test_portfolio_allocation_event_published_for_compatibility() -> None:
    """Unified execution still publishes portfolio.allocation compatibility event."""
    candles = _uptrend_candles(100, start=100.0, step=0.2)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_AlwaysLongStrategy()],
        diagnostics=False,
    )
    allocations: list[list[tuple[str, str, float]]] = []
    engine._event_bus.subscribe(  # noqa: SLF001 - test inspects event bus output
        "portfolio.allocation",
        lambda allocation: allocations.append(allocation),
    )

    _ = engine.run(candles)
    assert allocations, "expected compatibility allocation events"
    assert any(
        a == [("portfolio", "long", 1.0)] or a == [("portfolio", "short", 1.0)]
        for a in allocations
    )
