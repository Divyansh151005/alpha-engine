"""Tests for funding contrarian + price confirmation."""

from __future__ import annotations

import pytest

from alpha_engine.alpha import AlphaEngine
from alpha_engine.alpha.funding_contrarian_alpha import FundingContrarianAlphaStrategy
from alpha_engine.alpha.funding_price_confirm_alpha import (
    FundingPriceConfirmContrarianAlphaStrategy,
)
from alpha_engine.alpha.price_divergence_confirm_alpha import (
    PriceDivergenceContrarianConfirmAlphaStrategy,
)
from alpha_engine.core.event_bus import EventBus
from alpha_engine.state_engine.state_models import MarketStateVector


def _state(
    ts: int,
    *,
    funding: float,
    ret10: float,
    rolling_return_20: float = 0.10,
    symbol: str | None = None,
) -> MarketStateVector:
    return MarketStateVector(
        timestamp=ts,
        trend_state=0.5,
        volatility_state=0.5,
        liquidity_stress=0.5,
        momentum_pressure=0.5,
        compression_state=0.5,
        funding_rate=funding,
        rolling_return_10=ret10,
        rolling_return_20=rolling_return_20,
        symbol=symbol,
    )


def _warm_funding_extreme(
    strat: FundingPriceConfirmContrarianAlphaStrategy,
    *,
    last_rate: float,
    baseline: float = 0.0001,
    n_tail: int = 20,
) -> None:
    for _ in range(n_tail - 1):
        strat.note_funding(baseline)
    strat.note_funding(last_rate)


def test_confirm_blocks_long_when_price_not_weak() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(z_window=z, min_funding_score=0.01)
    # Strong negative tail → positive contrarian score
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=0.02)
    s.on_market_state(st)
    assert s.generate_signal(st) is None


def test_dual_alpha_engine_and_gate_requires_price_confirm() -> None:
    """Funding leg alone is insufficient when PriceDivergence confirm is registered."""
    bus = EventBus()
    z = 20
    fund = FundingContrarianAlphaStrategy(z_window=z)
    for _ in range(z - 1):
        fund.note_funding(0.0001)
    fund.note_funding(-0.002)
    confirm = PriceDivergenceContrarianConfirmAlphaStrategy(
        z_window=z, funding=fund, min_funding_score=0.01
    )
    engine = AlphaEngine(
        event_bus=bus,
        strategies=[fund, confirm],
        alpha_visibility=False,
        min_signal_keep_threshold=0.0,
    )
    st_block = _state(1, funding=-0.002, ret10=0.02)
    bus.publish("market.state", st_block)
    assert abs(engine.smoothed_signal) < 1e-9

    st_pass = _state(2, funding=-0.002, ret10=-0.02)
    bus.publish("market.state", st_pass)
    assert abs(engine.smoothed_signal) > 1e-6


def test_signal_debug_flow_increments_on_full_path() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=z,
        min_funding_score=0.01,
        signal_debug_flow=True,
    )
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=-0.02)
    s.on_market_state(st)
    s.reset_signal_debug_stats()
    assert s.generate_signal(st) is not None
    assert s.debug_stats["total_bars"] == 1
    assert s.debug_stats["regime_pass"] == 1
    assert s.debug_stats["funding_pass"] == 1
    assert s.debug_stats["price_confirm_pass"] == 1
    assert s.debug_stats["final_signals"] == 1


def test_confirm_passes_long_when_price_weak() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(z_window=z, min_funding_score=0.01)
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=-0.02)
    s.on_market_state(st)
    sig = s.generate_signal(st)
    assert sig is not None
    assert sig.score > 0.0
    assert sig.strategy_name == "FundingPriceConfirmContrarianAlphaStrategy"


def test_confirm_blocks_short_when_price_not_strong() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(z_window=z, min_funding_score=0.01)
    _warm_funding_extreme(s, last_rate=0.002, baseline=-0.0001, n_tail=z)
    st = _state(2, funding=0.002, ret10=-0.02)
    s.on_market_state(st)
    assert s.generate_signal(st) is None


def test_confirm_passes_short_when_price_strong() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(z_window=z, min_funding_score=0.01)
    _warm_funding_extreme(s, last_rate=0.002, baseline=-0.0001, n_tail=z)
    st = _state(2, funding=0.002, ret10=0.02)
    s.on_market_state(st)
    sig = s.generate_signal(st)
    assert sig is not None
    assert sig.score < 0.0


def test_below_min_funding_score_returns_none() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(z_window=z, min_funding_score=0.99)
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=-0.02)
    s.on_market_state(st)
    assert s.generate_signal(st) is None


def test_asset_disabled_skips_signal() -> None:
    z = 20
    cfg = {"SOL/USDT": {"enabled": False}}
    s = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=z, min_funding_score=0.01, asset_config=cfg
    )
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=-0.02, symbol="SOL/USDT")
    s.on_market_state(st)
    assert s.generate_signal(st) is None


def test_per_asset_min_funding_score_override() -> None:
    z = 20
    cfg = {"TIGHT/USDT": {"min_funding_score": 0.99}}
    s = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=z, min_funding_score=0.01, asset_config=cfg
    )
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st_tight = _state(1, funding=-0.002, ret10=-0.02, symbol="TIGHT/USDT")
    st_loose = _state(1, funding=-0.002, ret10=-0.02, symbol="LOOSE/USDT")
    s.on_market_state(st_tight)
    assert s.generate_signal(st_tight) is None
    s.on_market_state(st_loose)
    assert s.generate_signal(st_loose) is not None


def test_regime_gate_blocks_when_rr20_too_small() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=z,
        min_funding_score=0.01,
        regime_abs_return_20_min=0.05,
    )
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=-0.02, rolling_return_20=0.02)
    s.on_market_state(st)
    assert s.generate_signal(st) is None


def test_regime_gate_passes_when_rr20_large_enough() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=z,
        min_funding_score=0.01,
        regime_abs_return_20_min=0.05,
    )
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=-0.02, rolling_return_20=-0.08)
    s.on_market_state(st)
    assert s.generate_signal(st) is not None


def test_regime_band_blocks_when_rr20_above_max() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=z,
        min_funding_score=0.01,
        regime_abs_return_20_min=0.05,
        regime_abs_return_20_max=0.10,
    )
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=-0.02, rolling_return_20=-0.15)
    s.on_market_state(st)
    assert s.generate_signal(st) is None


def test_regime_band_passes_in_mid_range() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=z,
        min_funding_score=0.01,
        regime_abs_return_20_min=0.05,
        regime_abs_return_20_max=0.10,
    )
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=-0.02, rolling_return_20=-0.07)
    s.on_market_state(st)
    assert s.generate_signal(st) is not None


def test_regime_max_without_min_raises() -> None:
    with pytest.raises(ValueError, match="regime_abs_return_20_max requires"):
        FundingPriceConfirmContrarianAlphaStrategy(
            z_window=20,
            min_funding_score=0.01,
            regime_abs_return_20_max=0.1,
        )


def test_regime_max_le_min_raises() -> None:
    with pytest.raises(ValueError, match="regime_abs_return_20_max must be >"):
        FundingPriceConfirmContrarianAlphaStrategy(
            z_window=20,
            min_funding_score=0.01,
            regime_abs_return_20_min=0.05,
            regime_abs_return_20_max=0.05,
        )


def test_funding_regime_z_std_xor_raises() -> None:
    with pytest.raises(ValueError, match="funding_regime_z_abs_min and funding_regime_std_min"):
        FundingPriceConfirmContrarianAlphaStrategy(
            z_window=20,
            min_funding_score=0.01,
            funding_regime_z_abs_min=1.0,
        )


def test_funding_regime_adaptive_conflicts_with_static() -> None:
    with pytest.raises(ValueError, match="funding_regime_adaptive=True excludes"):
        FundingPriceConfirmContrarianAlphaStrategy(
            z_window=20,
            min_funding_score=0.01,
            funding_regime_adaptive=True,
            funding_regime_z_abs_min=1.0,
            funding_regime_std_min=1e-6,
        )


def test_funding_structure_blocks_when_z_threshold_too_high() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=z,
        min_funding_score=0.01,
        funding_regime_z_abs_min=500.0,
        funding_regime_std_min=1e-10,
        funding_regime_window=z,
    )
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=-0.02, rolling_return_20=-0.08)
    s.on_market_state(st)
    assert s.generate_signal(st) is None


def test_funding_structure_blocks_when_std_threshold_too_high() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=z,
        min_funding_score=0.01,
        funding_regime_z_abs_min=0.01,
        funding_regime_std_min=1.0,
        funding_regime_window=z,
    )
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=-0.02, rolling_return_20=-0.08)
    s.on_market_state(st)
    assert s.generate_signal(st) is None


def test_funding_structure_passes_with_moderate_thresholds() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=z,
        min_funding_score=0.01,
        funding_regime_z_abs_min=0.3,
        funding_regime_std_min=1e-7,
        funding_regime_window=z,
    )
    _warm_funding_extreme(s, last_rate=-0.002, baseline=0.0001, n_tail=z)
    st = _state(1, funding=-0.002, ret10=-0.02, rolling_return_20=-0.08)
    s.on_market_state(st)
    sig = s.generate_signal(st)
    assert sig is not None


def test_funding_persistence_blocks_mixed_sign_last_three() -> None:
    z = 20
    s = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=z,
        min_funding_score=0.01,
        funding_regime_z_abs_min=0.2,
        funding_regime_std_min=1e-7,
        funding_regime_window=z,
        funding_regime_same_sign_bars=3,
    )
    # Last 4 updates: strong +, weak +, weak +, negative → last 3 signs not all same
    rates = [0.0001] * (z - 4) + [0.004, 0.0002, 0.0002, -0.0001]
    s.note_funding_batch(rates)
    st = _state(1, funding=rates[-1], ret10=0.02, rolling_return_20=0.08)
    assert s.generate_signal(st) is None
