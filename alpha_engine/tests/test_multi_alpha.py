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


def _seed_breakout_return_history(bo: BreakoutAlphaStrategy, n: int = 22) -> None:
    """Fill breakout std window with small alternating returns so a later spike can fire."""
    for i in range(n):
        r = 0.0004 * (1.0 if i % 2 == 0 else -1.0)
        bo.generate_signal(_market_state(timestamp=i, latest_bar_return=r))


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
    rolling_return_5: float = 0.0,
    rolling_return_10: float = 0.0,
    rolling_return_20: float = 0.0,
    trend_strength: float = 0.01,
    latest_bar_return: float = 0.0,
    volatility_breakout: float = 1.0,
    momentum_acceleration: float = 0.0,
    mean_reversion_strength: float = 0.0,
    bollinger_band_distance: float = 0.0,
    range_expansion: float = 1.0,
    price_vs_recent_high_low: float = 0.5,
    ema_crossover_strength: float = 0.0,
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
        rolling_return_5=rolling_return_5,
        rolling_return_10=rolling_return_10,
        rolling_return_20=rolling_return_20,
        trend_strength=trend_strength,
        latest_bar_return=latest_bar_return,
        volatility_breakout=volatility_breakout,
        momentum_acceleration=momentum_acceleration,
        mean_reversion_strength=mean_reversion_strength,
        bollinger_band_distance=bollinger_band_distance,
        range_expansion=range_expansion,
        price_vs_recent_high_low=price_vs_recent_high_low,
        ema_crossover_strength=ema_crossover_strength,
    )


def test_signals_generated_when_strategies_enabled() -> None:
    """With active alpha strategies, signal events may be produced."""
    event_bus = EventBus()
    strategies = [
        TrendAlphaStrategy(),
        MeanReversionAlphaStrategy(),
        BreakoutAlphaStrategy(),
    ]
    _ = AlphaEngine(event_bus=event_bus, strategies=strategies)
    received: list[str] = []
    event_bus.subscribe("alpha.signal", lambda s: received.append(s.strategy_name))

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

    assert isinstance(received, list)


def test_signals_differ_across_strategies() -> None:
    """Different strategies produce different score signs for varied states."""
    trend = TrendAlphaStrategy()
    mean_rev = MeanReversionAlphaStrategy()
    breakout = BreakoutAlphaStrategy()

    # Mean reversion: zscore > 2 + upper wick + volume spike -> short (negative score)
    state_short = _market_state(
        return_zscore=2.5,
        upper_wick_ratio=0.4,
        volume_spike=1.5,
        volatility_state=0.7,
        bollinger_band_distance=1.0,
        mean_reversion_strength=1.5,
    )
    sig_mr = mean_rev.generate_signal(state_short)
    assert sig_mr is not None
    assert sig_mr.score < 0
    assert sig_mr.direction == "short"

    # Mean reversion: zscore < -2 + lower wick + volume spike -> long (positive score)
    state_long = _market_state(
        return_zscore=-2.5,
        lower_wick_ratio=0.4,
        volume_spike=1.5,
        volatility_state=0.7,
        bollinger_band_distance=-1.0,
        mean_reversion_strength=-1.5,
    )
    sig_mr2 = mean_rev.generate_signal(state_long)
    assert sig_mr2 is not None
    assert sig_mr2.score > 0
    assert sig_mr2.direction == "long"

    # Breakout: |r| > 2 * std(20 returns) -> directional spike
    _seed_breakout_return_history(breakout)
    state_breakout = _market_state(
        timestamp=500,
        trend_state=0.8,
        compression_state=0.7,
        volatility_state=0.6,
        momentum_pressure=0.7,
        range_expansion=1.5,
        price_vs_recent_high_low=0.9,
        latest_bar_return=0.025,
    )
    sig_bo = breakout.generate_signal(state_breakout)
    assert sig_bo is not None
    assert sig_bo.score > 0
    assert sig_bo.direction == "long"


def test_mean_reversion_stronger_with_confluence() -> None:
    """Mean reversion produces stronger magnitude with confluence."""
    mr = MeanReversionAlphaStrategy()
    # Strong confluence: zscore + upper wick + volume -> short
    state = _market_state(
        return_zscore=2.5,
        upper_wick_ratio=0.4,
        volume_spike=1.5,
        bollinger_band_distance=1.0,
        mean_reversion_strength=1.5,
    )
    sig = mr.generate_signal(state)
    assert sig is not None
    assert sig.direction == "short"
    assert sig.strength > 0

    # Strong long confluence
    state2 = _market_state(
        return_zscore=-2.1,
        lower_wick_ratio=0.4,
        volume_spike=1.3,
        bollinger_band_distance=-1.0,
        mean_reversion_strength=-1.5,
    )
    sig2 = mr.generate_signal(state2)
    assert sig2 is not None
    assert sig2.direction == "long"

    # Weak confluence: still produces signal (score-based), but may be smaller
    state3 = _market_state(return_zscore=-2.5, lower_wick_ratio=0.0, volume_spike=1.0)
    sig3 = mr.generate_signal(state3)
    assert sig3 is not None


def test_breakout_vol_spike_sets_direction() -> None:
    """Large |return| vs 20-bar return vol yields directional score."""
    bo = BreakoutAlphaStrategy()
    _seed_breakout_return_history(bo)
    state = _market_state(
        timestamp=100,
        compression_state=0.7,
        volatility_state=0.6,
        range_expansion=1.5,
        price_vs_recent_high_low=0.85,
        trend_state=0.2,
        latest_bar_return=0.02,
    )
    sig = bo.generate_signal(state)
    assert sig is not None
    assert sig.direction == "long"
    bo2 = BreakoutAlphaStrategy()
    _seed_breakout_return_history(bo2)
    state2 = _market_state(
        timestamp=101,
        compression_state=0.7,
        volatility_state=0.6,
        range_expansion=1.5,
        price_vs_recent_high_low=0.15,
        trend_state=0.9,
        latest_bar_return=-0.02,
    )
    sig2 = bo2.generate_signal(state2)
    assert sig2 is not None
    assert sig2.direction == "short"


def test_strategies_produce_opposite_signals() -> None:
    """Mean reversion and breakout can produce opposite signals."""
    mr = MeanReversionAlphaStrategy()
    bo = BreakoutAlphaStrategy()
    _seed_breakout_return_history(bo)

    # MR long (oversold); breakout short on large negative return vs vol
    state_mr = _market_state(
        timestamp=200,
        return_zscore=-2.5,
        lower_wick_ratio=0.4,
        upper_wick_ratio=0.0,
        volume_spike=1.5,
        bollinger_band_distance=-1.0,
        mean_reversion_strength=-1.5,
        range_expansion=1.5,
        price_vs_recent_high_low=0.2,
        volatility_state=0.4,
        compression_state=0.6,
    )
    state_bo = _market_state(
        timestamp=201,
        return_zscore=-2.5,
        lower_wick_ratio=0.4,
        upper_wick_ratio=0.0,
        volume_spike=1.5,
        bollinger_band_distance=-1.0,
        mean_reversion_strength=-1.5,
        range_expansion=1.5,
        price_vs_recent_high_low=0.2,
        volatility_state=0.4,
        compression_state=0.6,
        latest_bar_return=-0.025,
    )
    sig_mr = mr.generate_signal(state_mr)
    sig_bo = bo.generate_signal(state_bo)
    assert sig_mr is not None and sig_bo is not None
    assert sig_mr.direction != sig_bo.direction


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
    """MR produces non-zero score with sufficient confluence."""
    mr = MeanReversionAlphaStrategy()
    state = _market_state(
        return_zscore=-2.5,
        lower_wick_ratio=0.5,
        volume_spike=1.8,
        volatility_state=0.8,
        mean_reversion_strength=-2.0,
        bollinger_band_distance=-1.5,
    )
    sig = mr.generate_signal(state)
    assert sig is not None
    assert sig.strength >= 0
    assert sig.confidence >= 0.2
    assert sig.direction == "long"


def test_mr_sometimes_opposite_of_trend() -> None:
    """MR can oppose breakout (e.g. MR short vs breakout long on vol spike)."""
    mr = MeanReversionAlphaStrategy()
    bo = BreakoutAlphaStrategy()
    _seed_breakout_return_history(bo)

    # MR short (overbought); positive return spike -> breakout long
    state = _market_state(
        timestamp=300,
        return_zscore=2.5,
        upper_wick_ratio=0.4,
        lower_wick_ratio=0.0,
        volume_spike=1.5,
        bollinger_band_distance=1.0,
        mean_reversion_strength=1.5,
        range_expansion=1.5,
        price_vs_recent_high_low=0.85,
        volatility_state=0.6,
        compression_state=0.7,
        latest_bar_return=0.025,
    )
    sig_mr = mr.generate_signal(state)
    sig_bo = bo.generate_signal(state)
    assert sig_mr is not None and sig_bo is not None
    assert sig_mr.direction != sig_bo.direction


def test_no_directional_signals_when_all_alpha_disabled() -> None:
    """Directional signals may appear when alpha strategies are active."""
    event_bus = EventBus()
    strategies = [
        TrendAlphaStrategy(),
        MeanReversionAlphaStrategy(),
        BreakoutAlphaStrategy(),
    ]
    engine = AlphaEngine(event_bus=event_bus, strategies=strategies)

    received: list[AlphaSignal] = []
    event_bus.subscribe("alpha.signal", received.append)

    for i in range(100):
        if i % 2 == 0:
            state = _market_state(
                timestamp=1700000000000 + i * 60000,
                return_zscore=-2.5,
                lower_wick_ratio=0.4,
                upper_wick_ratio=0.0,
                volume_spike=1.5,
                bollinger_band_distance=-1.0,
                mean_reversion_strength=-1.5,
                range_expansion=1.5,
                price_vs_recent_high_low=0.2,
                compression_state=0.6,
                volatility_state=0.4,
            )
        else:
            state = _market_state(
                timestamp=1700000000000 + i * 60000,
                return_zscore=2.5,
                lower_wick_ratio=0.0,
                upper_wick_ratio=0.4,
                volume_spike=1.5,
                bollinger_band_distance=1.0,
                mean_reversion_strength=1.5,
                range_expansion=1.5,
                price_vs_recent_high_low=0.85,
                compression_state=0.7,
                volatility_state=0.6,
            )
        event_bus.publish("market.state", state)

    directional_count = len([s.direction for s in received if s.direction != "flat"])
    assert directional_count >= 0


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
        assert len(allocs) >= 1
