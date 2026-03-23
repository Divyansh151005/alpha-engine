"""Tests for feature computation modules and event flow."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_engine.core.event_bus import EventBus
from alpha_engine.data.models import OHLCV
from alpha_engine.features.feature_engine import FeatureEngine, FeatureVector
from alpha_engine.features.microstructure import lower_wick_ratio, upper_wick_ratio
from alpha_engine.features.trend import ema
from alpha_engine.features.volatility import atr, calculate_returns, realized_volatility


def _make_candle(i: int) -> OHLCV:
    close = 100.0 + (i * 0.5)
    return OHLCV(
        symbol="BTC/USDT",
        timeframe="1m",
        timestamp=1700000000000 + (i * 60000),
        open=close - 0.2,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=10.0 + (i % 5),
    )


def test_volatility_calculations() -> None:
    prices = np.array([100.0, 101.0, 103.0, 102.0], dtype=np.float64)
    returns = calculate_returns(prices)

    np.testing.assert_allclose(
        returns,
        np.array([0.01, 2.0 / 101.0, -1.0 / 103.0], dtype=np.float64),
    )

    vol = realized_volatility(returns, window=3)
    expected = float(np.sqrt(np.mean(np.square(returns[-3:]))) * np.sqrt(3))
    assert vol == expected


def test_ema_correctness() -> None:
    series = [10.0, 12.0, 14.0]
    # alpha = 2 / (2 + 1) = 2/3
    # ema1 = 10
    # ema2 = (2/3)*12 + (1/3)*10 = 11.333333...
    # ema3 = (2/3)*14 + (1/3)*11.333333... = 13.111111...
    value = ema(series, period=2)
    assert value == pytest.approx(13.11111111111111)


def test_atr_calculation() -> None:
    high = [10.0, 12.0, 13.0]
    low = [8.0, 9.0, 10.0]
    close = [9.0, 11.0, 12.0]

    # TRs: [2, max(3,3,0)=3, max(3,2,1)=3]
    result = atr(high, low, close, window=3)
    assert result == pytest.approx(8.0 / 3.0)


def test_feature_vector_generation() -> None:
    bus = EventBus()
    _ = FeatureEngine(event_bus=bus)

    received: list[FeatureVector] = []
    bus.subscribe("market.features", lambda fv: received.append(fv))

    total = FeatureEngine.EMA_LONG_PERIOD + 5
    for i in range(total):
        bus.publish("market.candle", _make_candle(i))

    assert received, "expected at least one feature vector"
    latest = received[-1]
    assert latest.timestamp == _make_candle(total - 1).timestamp
    assert np.isfinite(latest.realized_vol)
    assert np.isfinite(latest.atr)
    assert np.isfinite(latest.trend_strength)
    assert np.isfinite(latest.momentum)
    assert np.isfinite(latest.range_compression)
    assert np.isfinite(latest.volume_spike)
    assert np.isfinite(latest.return_zscore)
    assert 0.0 <= latest.lower_wick_ratio <= 1.0
    assert 0.0 <= latest.upper_wick_ratio <= 1.0


def test_wick_ratio_calculation() -> None:
    """Lower/upper wick ratios are correctly computed from OHLC."""
    # Candle: O=100, H=105, L=95, C=102. Range=10. Body 100-102. Lower wick 95-100=5, upper 102-105=3
    assert lower_wick_ratio(100.0, 105.0, 95.0, 102.0) == pytest.approx(0.5)
    assert upper_wick_ratio(100.0, 105.0, 95.0, 102.0) == pytest.approx(0.3)
    # Doji: O=C=100, H=101, L=99. Lower=1/2=0.5, upper=1/2=0.5
    assert lower_wick_ratio(100.0, 101.0, 99.0, 100.0) == pytest.approx(0.5)
    assert upper_wick_ratio(100.0, 101.0, 99.0, 100.0) == pytest.approx(0.5)


def test_event_published_when_candle_received() -> None:
    bus = EventBus()
    _ = FeatureEngine(event_bus=bus)

    published_count = 0

    def _capture(_: FeatureVector) -> None:
        nonlocal published_count
        published_count += 1

    bus.subscribe("market.features", _capture)

    for i in range(FeatureEngine.EMA_LONG_PERIOD + 1):
        bus.publish("market.candle", _make_candle(i))

    assert published_count >= 1
