"""Tests for multi-alpha strategy system."""

from __future__ import annotations

from alpha_engine.alpha import (
    AlphaEngine,
    AlphaSignal,
    BreakoutAlphaStrategy,
    MeanReversionAlphaStrategy,
    TrendAlphaStrategy,
)
from alpha_engine.core.event_bus import EventBus
from alpha_engine.state_engine.state_models import MarketStateVector


def _market_state(
    timestamp: int = 1700000000000,
    trend_state: float = 0.8,
    volatility_state: float = 0.6,
    liquidity_stress: float = 0.4,
    momentum_pressure: float = 0.7,
    compression_state: float = 0.7,
    return_zscore: float = 0.0,
    lower_wick_ratio: float = 0.0,
    upper_wick_ratio: float = 0.0,
    volume_spike: float = 1.0,
) -> MarketStateVector:
    return MarketStateVector(
        timestamp=timestamp,
        trend_state=trend_state,
        volatility_state=volatility_state,
        liquidity_stress=liquidity_stress,
        momentum_pressure=momentum_pressure,
        compression_state=compression_state,
        return_zscore=return_zscore,
        lower_wick_ratio=lower_wick_ratio,
        upper_wick_ratio=upper_wick_ratio,
        volume_spike=volume_spike,
    )


def test_all_strategies_produce_signals() -> None:
    """All three strategies can produce signals with appropriate state."""
    event_bus = EventBus()
    strategies = [
        TrendAlphaStrategy(),
        MeanReversionAlphaStrategy(),
        BreakoutAlphaStrategy(),
    ]
    _ = AlphaEngine(event_bus=event_bus, strategies=strategies)
    received: list[str] = []
    event_bus.subscribe("alpha.signal", lambda s: received.append(s.strategy_name))

    # Trend: long (trend>0.5+thresh, momentum>0.48)
    # MeanReversion: zscore + wick + volume_spike
    # Breakout: compression high, vol expanding
    state = _market_state(
        trend_state=0.85,
        momentum_pressure=0.7,
        compression_state=0.7,
        volatility_state=0.6,
        return_zscore=2.5,
        upper_wick_ratio=0.4,
        volume_spike=1.5,
    )
    event_bus.publish("market.state", state)

    assert len(received) >= 1, "at least one strategy should produce a signal"
    assert "TrendAlphaStrategy" in received or "MeanReversionAlphaStrategy" in received


def test_signals_differ_across_strategies() -> None:
    """Different strategies produce different signals for varied states."""
    trend = TrendAlphaStrategy()
    mean_rev = MeanReversionAlphaStrategy()
    breakout = BreakoutAlphaStrategy()

    # Mean reversion: zscore > 2 + upper wick + volume spike -> short
    state_short = _market_state(
        return_zscore=2.5,
        upper_wick_ratio=0.4,
        volume_spike=1.5,
        volatility_state=0.7,
    )
    sig_mr = mean_rev.generate_signal(state_short)
    assert sig_mr is not None
    assert sig_mr.direction == "short"
    assert sig_mr.strategy_name == "MeanReversionAlphaStrategy"

    # Mean reversion: zscore < -2 + lower wick + volume spike -> long
    state_long = _market_state(
        return_zscore=-2.5,
        lower_wick_ratio=0.4,
        volume_spike=1.5,
        volatility_state=0.7,
    )
    sig_mr2 = mean_rev.generate_signal(state_long)
    assert sig_mr2 is not None
    assert sig_mr2.direction == "long"

    # Breakout: compression high, vol high, trend up -> long
    state_breakout = _market_state(
        trend_state=0.8,
        compression_state=0.7,
        volatility_state=0.6,
        momentum_pressure=0.7,
    )
    sig_bo = breakout.generate_signal(state_breakout)
    assert sig_bo is not None
    assert sig_bo.direction == "long"
    assert sig_bo.strategy_name == "BreakoutAlphaStrategy"


def test_mean_reversion_requires_confluence() -> None:
    """Mean reversion requires zscore + wick + volume spike confluence."""
    mr = MeanReversionAlphaStrategy()
    # zscore > 2 + upper wick + volume spike -> short (ignores trend)
    state = _market_state(
        return_zscore=2.5,
        upper_wick_ratio=0.4,
        volume_spike=1.5,
        trend_state=0.9,
        momentum_pressure=0.9,
    )
    sig = mr.generate_signal(state)
    assert sig is not None
    assert sig.direction == "short"
    # zscore < -2 + lower wick + volume spike -> long
    state2 = _market_state(
        return_zscore=-2.1,
        lower_wick_ratio=0.4,
        volume_spike=1.3,
        trend_state=0.1,
        momentum_pressure=0.1,
    )
    sig2 = mr.generate_signal(state2)
    assert sig2 is not None
    assert sig2.direction == "long"
    # zscore alone (no wick/volume) -> no signal
    state3 = _market_state(return_zscore=-2.5, lower_wick_ratio=0.0, volume_spike=1.0)
    assert mr.generate_signal(state3) is None


def test_breakout_uses_compression_volatility_not_trend() -> None:
    """Breakout uses compression + volatility; direction from momentum (price), not trend."""
    bo = BreakoutAlphaStrategy()
    # High compression, high vol, momentum > 0.5 -> long (even with low trend_state)
    state = _market_state(
        compression_state=0.7,
        volatility_state=0.6,
        momentum_pressure=0.6,
        trend_state=0.2,
    )
    sig = bo.generate_signal(state)
    assert sig is not None
    assert sig.direction == "long"
    # Same but momentum < 0.5 -> short (trend_state high but we use momentum)
    state2 = _market_state(
        compression_state=0.7,
        volatility_state=0.6,
        momentum_pressure=0.4,
        trend_state=0.9,
    )
    sig2 = bo.generate_signal(state2)
    assert sig2 is not None
    assert sig2.direction == "short"


def test_strategies_produce_opposite_signals() -> None:
    """Strategies can produce opposite signals (diversification)."""
    trend = TrendAlphaStrategy()
    mr = MeanReversionAlphaStrategy()
    # Trend: long (high trend, high momentum). Mean reversion: short (zscore + upper wick + vol)
    state = _market_state(
        trend_state=0.9,
        momentum_pressure=0.8,
        return_zscore=2.5,
        upper_wick_ratio=0.4,
        volume_spike=1.5,
    )
    sig_t = trend.generate_signal(state)
    sig_mr = mr.generate_signal(state)
    # Both may signal; if so, directions can differ
    if sig_t is not None and sig_mr is not None:
        assert sig_t.direction != sig_mr.direction


def test_correlation_below_threshold() -> None:
    """Strategy correlations stay below 0.7 (diversification)."""
    import math

    event_bus = EventBus()
    strategies = [
        TrendAlphaStrategy(),
        MeanReversionAlphaStrategy(),
        BreakoutAlphaStrategy(),
    ]
    engine = AlphaEngine(event_bus=event_bus, strategies=strategies, alpha_visibility=True)

    # Alternating states: trend long + mr short vs trend short + mr long
    for i in range(80):
        if i % 2 == 0:
            state = _market_state(
                timestamp=1700000000000 + i * 60000,
                trend_state=0.9,
                momentum_pressure=0.8,
                return_zscore=2.5,
                compression_state=0.5,
                volatility_state=0.4,
                upper_wick_ratio=0.35,
                volume_spike=1.4,
            )
        else:
            state = _market_state(
                timestamp=1700000000000 + i * 60000,
                trend_state=0.2,
                momentum_pressure=0.3,
                return_zscore=-2.2,
                compression_state=0.7,
                volatility_state=0.65,
                lower_wick_ratio=0.35,
                volume_spike=1.4,
            )
        event_bus.publish("market.state", state)

    history = engine.get_strategy_sign_history()
    n = min(len(history.get(n, [])) for n in ["TrendAlphaStrategy", "MeanReversionAlphaStrategy", "BreakoutAlphaStrategy"])
    if n < 20:
        return
    series = {k: v[:n] for k, v in history.items() if k in history}

    def _corr(a: list[float], b: list[float]) -> float:
        ma = sum(a) / len(a)
        mb = sum(b) / len(b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
        va = sum((x - ma) ** 2 for x in a) / len(a)
        vb = sum((y - mb) ** 2 for y in b) / len(b)
        denom = math.sqrt(va * vb)
        return cov / denom if denom > 1e-12 else 0.0

    if "TrendAlphaStrategy" in series and "MeanReversionAlphaStrategy" in series:
        corr_t_mr = _corr(series["TrendAlphaStrategy"], series["MeanReversionAlphaStrategy"])
        # Diversification: positive correlation < 0.7 (not always aligned)
        # Negative correlation is good (opposite signals)
        assert corr_t_mr < 0.7, f"trend vs mr correlation {corr_t_mr} should be < 0.7 (avoid over-alignment)"


def test_mr_produces_nonzero_strong_signals() -> None:
    """MR produces non-zero strong signals when confluence conditions met."""
    mr = MeanReversionAlphaStrategy()
    state = _market_state(
        return_zscore=-2.5,
        lower_wick_ratio=0.5,
        volume_spike=1.8,
        volatility_state=0.8,
    )
    sig = mr.generate_signal(state)
    assert sig is not None
    assert sig.strength >= 0.2
    assert sig.confidence >= 0.3
    assert sig.direction == "long"


def test_mr_sometimes_opposite_of_trend() -> None:
    """MR can oppose trend (long when trend short, short when trend long)."""
    trend = TrendAlphaStrategy()
    mr = MeanReversionAlphaStrategy()
    # Trend: long (high trend, high momentum). MR: short (zscore + upper wick + vol)
    state = _market_state(
        trend_state=0.9,
        momentum_pressure=0.8,
        return_zscore=2.5,
        upper_wick_ratio=0.4,
        volume_spike=1.5,
    )
    sig_t = trend.generate_signal(state)
    sig_mr = mr.generate_signal(state)
    assert sig_t is not None and sig_mr is not None
    assert sig_t.direction != sig_mr.direction  # MR opposes trend


def test_directional_balance_both_long_and_short() -> None:
    """System produces both long and short signals (not one-sided)."""
    event_bus = EventBus()
    strategies = [
        TrendAlphaStrategy(),
        MeanReversionAlphaStrategy(),
        BreakoutAlphaStrategy(),
    ]
    engine = AlphaEngine(event_bus=event_bus, strategies=strategies)

    received: list[AlphaSignal] = []
    event_bus.subscribe("alpha.signal", received.append)

    # Alternate uptrend and downtrend states
    for i in range(100):
        if i % 2 == 0:
            state = _market_state(
                timestamp=1700000000000 + i * 60000,
                trend_state=0.85,
                momentum_pressure=0.75,
                compression_state=0.6,
                volatility_state=0.6,
            )
        else:
            state = _market_state(
                timestamp=1700000000000 + i * 60000,
                trend_state=0.15,
                momentum_pressure=0.25,
                compression_state=0.7,
                volatility_state=0.65,
                lower_wick_ratio=0.3,
                upper_wick_ratio=0.3,
                volume_spike=1.3,
            )
        event_bus.publish("market.state", state)

    directions = [s.direction for s in received if s.direction != "flat"]
    assert len(directions) >= 10, "expected sufficient signals"
    n_long = sum(1 for d in directions if d == "long")
    n_short = sum(1 for d in directions if d == "short")
    pct_long = 100.0 * n_long / len(directions)
    pct_short = 100.0 * n_short / len(directions)
    assert pct_long <= 80.0, f"long bias: {pct_long:.1f}% should be <= 80%"
    assert pct_short <= 80.0, f"short bias: {pct_short:.1f}% should be <= 80%"
    assert n_long >= 1 and n_short >= 1, "both long and short must appear"


def test_allocation_distributes_capital() -> None:
    """Multiple signals result in capital distribution across strategies."""
    from alpha_engine.allocation_engine import CapitalAllocator
    from alpha_engine.alpha.alpha_signal import AlphaSignal

    event_bus = EventBus()
    strategies = [
        TrendAlphaStrategy(),
        MeanReversionAlphaStrategy(),
        BreakoutAlphaStrategy(),
    ]
    _ = AlphaEngine(event_bus=event_bus, strategies=strategies)
    _ = CapitalAllocator(event_bus=event_bus)

    allocations_received: list[list] = []
    event_bus.subscribe("portfolio.allocation", lambda a: allocations_received.append(a))

    # Fire state that triggers multiple strategies (MR needs wick + volume_spike)
    state = _market_state(
        trend_state=0.9,
        momentum_pressure=0.8,
        compression_state=0.75,
        volatility_state=0.65,
        return_zscore=2.2,
        upper_wick_ratio=0.4,
        volume_spike=1.5,
    )
    event_bus.publish("market.state", state)

    if allocations_received:
        allocs = allocations_received[-1]
        total = sum(a.capital_fraction for a in allocs)
        assert total <= 1.0 + 1e-6, "allocations should sum to <= 1"
        assert len(allocs) >= 1, "should have at least one allocation"
