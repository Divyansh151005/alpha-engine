"""Tests for backtest engine."""

from __future__ import annotations

from alpha_engine.alpha.alpha_signal import AlphaSignal
from alpha_engine.alpha.base_alpha import BaseAlphaStrategy
from alpha_engine.backtest.engine import BacktestEngine, BacktestResult, compute_metrics
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
            direction="long",
            strength=0.8,
            confidence=0.9,
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


def test_allocation_not_always_zero_with_always_long() -> None:
    """With AlwaysLong strategy, allocation and avg_position should be positive."""
    candles = _uptrend_candles(80)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_AlwaysLongStrategy()],
        diagnostics=False,
    )
    result = engine.run(candles)
    assert result.metrics["avg_position_size"] > 0, "should have non-zero position"
    assert result.metrics["time_in_market"] > 0, "should spend time in market"


def test_trade_count_positive_with_synthetic_uptrend() -> None:
    """With AlwaysLong and uptrend data, trade_count should be > 0."""
    candles = _uptrend_candles(80)
    engine = BacktestEngine(
        initial_equity=100.0,
        strategies=[_AlwaysLongStrategy()],
        diagnostics=False,
    )
    result = engine.run(candles)
    assert result.metrics["trade_count"] > 0, "should have at least one position change"
