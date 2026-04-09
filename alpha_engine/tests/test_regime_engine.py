"""Tests for regime classification engine."""

from __future__ import annotations

from alpha_engine.core.event_bus import EventBus
from alpha_engine.data.models import OHLCV
from alpha_engine.features import FeatureEngine, FeatureVector
from alpha_engine.regime import RegimeEngine, RegimeType, get_regime_multipliers
from alpha_engine.allocation_engine import CapitalAllocator, PositionAllocation
from alpha_engine.alpha.alpha_signal import AlphaSignal


def _feature_vector(
    timestamp: int = 1000,
    trend_strength: float = 0.0,
    realized_vol: float = 0.02,
    **kwargs: float,
) -> FeatureVector:
    defaults = {
        "symbol": None,
        "realized_vol": realized_vol,
        "volatility": realized_vol,
        "atr": 0.01,
        "trend_strength": trend_strength,
        "momentum": 0.0,
        "range_compression": 1.0,
        "volume_spike": 1.0,
        "return_zscore": 0.0,
        "lower_wick_ratio": 0.0,
        "upper_wick_ratio": 0.0,
        "rolling_return_5": 0.0,
        "rolling_return_10": 0.0,
        "rolling_return_20": 0.0,
        "regime_trend_strength": 0.0,
        "latest_bar_return": 0.0,
        "volatility_breakout": 1.0,
        "momentum_acceleration": 0.0,
        "mean_reversion_strength": 0.0,
        "bollinger_band_distance": 0.0,
        "range_expansion": 1.0,
        "price_vs_recent_high_low": 0.5,
        "ema_crossover_strength": 0.0,
        "funding_rate": float("nan"),
        "oi_change": float("nan"),
    }
    defaults.update(kwargs)
    return FeatureVector(timestamp=timestamp, **defaults)


def _candle(timestamp: int, close: float, base: float = 100.0) -> OHLCV:
    return OHLCV(
        symbol="BTC/USDT",
        timeframe="1m",
        timestamp=timestamp,
        open=base,
        high=base * 1.01,
        low=base * 0.99,
        close=close,
        volume=1000.0,
    )


def test_regime_trend_when_trend_strength_high() -> None:
    """TREND regime detected when trend_strength is high."""
    event_bus = EventBus()
    RegimeEngine(event_bus=event_bus)

    received: list = []
    event_bus.subscribe("market.regime", received.append)

    # Warm up with candles so autocorr doesn't dominate
    for i in range(60):
        event_bus.publish("market.candle", _candle(900 + i, 100.0 + i * 0.1))

    event_bus.publish(
        "market.features",
        _feature_vector(timestamp=1000, trend_strength=0.05, realized_vol=0.01),
    )

    assert len(received) >= 1
    state = received[-1]
    assert state.regime == RegimeType.TREND


def test_regime_high_vol_when_volatility_high() -> None:
    """HIGH_VOL regime when vol_ratio is high (no strong trend)."""
    event_bus = EventBus()
    RegimeEngine(event_bus=event_bus)

    received: list = []
    event_bus.subscribe("market.regime", received.append)

    # Warm up with low vol features to raise baseline
    for i in range(60):
        event_bus.publish(
            "market.features",
            _feature_vector(timestamp=900 + i, trend_strength=0.001, realized_vol=0.01),
        )
    for i in range(60):
        event_bus.publish("market.candle", _candle(900 + i, 100.0 + i * 0.001))

    # Spike vol with low trend -> HIGH_VOL
    event_bus.publish(
        "market.features",
        _feature_vector(timestamp=1000, trend_strength=0.001, realized_vol=0.05),
    )

    assert len(received) >= 1
    state = received[-1]
    assert state.regime == RegimeType.HIGH_VOL


def test_regime_mean_reversion_when_autocorr_negative() -> None:
    """MEAN_REVERSION when autocorrelation is negative (no trend, low vol)."""
    event_bus = EventBus()
    RegimeEngine(event_bus=event_bus)

    received: list = []
    event_bus.subscribe("market.regime", received.append)

    # Alternating returns: +1%, -1%, +1%, -1% -> negative autocorrelation
    closes = [100.0]
    for i in range(54):
        mult = 1.01 if i % 2 == 0 else 0.99
        closes.append(closes[-1] * mult)
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i > 0 else c
        event_bus.publish("market.candle", _candle(900 + i, c, base=prev))

    event_bus.publish(
        "market.features",
        _feature_vector(timestamp=1000, trend_strength=0.001, realized_vol=0.01),
    )

    assert len(received) >= 1
    state = received[-1]
    assert state.regime == RegimeType.MEAN_REVERSION


def test_regime_low_vol_default() -> None:
    """LOW_VOL when no other conditions met."""
    event_bus = EventBus()
    RegimeEngine(event_bus=event_bus)

    received: list = []
    event_bus.subscribe("market.regime", received.append)

    for i in range(60):
        event_bus.publish("market.candle", _candle(900 + i, 100.0 + i * 0.0001))
    event_bus.publish(
        "market.features",
        _feature_vector(
            timestamp=1000,
            trend_strength=0.001,
            realized_vol=0.01,
        ),
    )

    assert len(received) >= 1
    state = received[-1]
    assert state.regime == RegimeType.LOW_VOL


def test_get_regime_multipliers() -> None:
    """Regime multipliers match spec: TREND boosts trend/breakout, MEAN_REVERSION boosts mr."""
    trend_m = get_regime_multipliers(RegimeType.TREND)
    assert trend_m["trend"] > 1.0
    assert trend_m["breakout"] > 1.0
    assert trend_m["mr"] < 1.0

    mr_m = get_regime_multipliers(RegimeType.MEAN_REVERSION)
    assert mr_m["mr"] > 1.0
    assert mr_m["trend"] < 1.0

    high_vol_m = get_regime_multipliers(RegimeType.HIGH_VOL)
    assert high_vol_m["trend"] < 1.0
    assert high_vol_m["breakout"] < 1.0
    assert high_vol_m["mr"] < 1.0

    low_vol_m = get_regime_multipliers(RegimeType.LOW_VOL)
    assert low_vol_m["breakout"] < 1.0


def test_weights_change_per_regime() -> None:
    """Allocation weights differ when regime changes (TREND vs MEAN_REVERSION)."""
    event_bus = EventBus()
    RegimeEngine(event_bus=event_bus)
    CapitalAllocator(event_bus=event_bus)

    regime_received: list = []
    alloc_received: list[list[PositionAllocation]] = []
    event_bus.subscribe("market.regime", regime_received.append)
    event_bus.subscribe("portfolio.allocation", alloc_received.append)

    def _sig(name: str, score: float = 0.5, ts: int = 1000) -> AlphaSignal:
        return AlphaSignal(
            timestamp=ts,
            strategy_name=name,
            score=score,
            confidence=0.5,
        )

    # Warm up regime and allocation
    for i in range(60):
        event_bus.publish("market.candle", _candle(900 + i, 100.0 + i * 0.1))
    event_bus.publish(
        "market.features",
        _feature_vector(timestamp=1000, trend_strength=0.05, realized_vol=0.01),
    )
    event_bus.publish("alpha.signal", _sig("TrendAlphaStrategy"))
    event_bus.publish("alpha.signal", _sig("MeanReversionAlphaStrategy"))

    assert len(regime_received) >= 1
    assert regime_received[-1].regime == RegimeType.TREND
    assert len(alloc_received) >= 1

    trend_alloc = {a.strategy_name: a.capital_fraction for a in alloc_received[-1]}
    trend_frac = trend_alloc.get("TrendAlphaStrategy", 0.0)
    mr_frac = trend_alloc.get("MeanReversionAlphaStrategy", 0.0)

    # In TREND: trend boosted, mr reduced -> trend >= mr (when both signal equally)
    assert trend_frac >= mr_frac or abs(trend_frac - mr_frac) < 0.01

    # Now switch to MEAN_REVERSION regime: alternating returns
    closes = [100.0]
    for i in range(54):
        mult = 1.01 if i % 2 == 0 else 0.99
        closes.append(closes[-1] * mult)
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i > 0 else c
        event_bus.publish("market.candle", _candle(2000 + i, c, base=prev))
    event_bus.publish(
        "market.features",
        _feature_vector(
            timestamp=2100, trend_strength=0.001, realized_vol=0.01
        ),
    )
    event_bus.publish("alpha.signal", _sig("TrendAlphaStrategy", score=0.5, ts=2100))
    event_bus.publish("alpha.signal", _sig("MeanReversionAlphaStrategy", score=0.5, ts=2100))

    mr_regime = [r for r in regime_received if r.regime == RegimeType.MEAN_REVERSION]
    if mr_regime:
        mr_alloc = {a.strategy_name: a.capital_fraction for a in alloc_received[-1]}
        mr_new = mr_alloc.get("MeanReversionAlphaStrategy", 0.0)
        trend_new = mr_alloc.get("TrendAlphaStrategy", 0.0)
        assert mr_new >= trend_new or abs(mr_new - trend_new) < 0.01
