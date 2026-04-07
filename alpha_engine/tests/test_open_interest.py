"""Tests for open-interest fetch helpers (alignment and change series)."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_engine.data.models import OHLCV
from alpha_engine.data.open_interest import (
    align_open_interest_to_timestamps,
    attach_open_interest_to_ohlcv,
    get_oi_coverage_range,
    oi_change_series,
    open_interest_period_for_candle_timeframe,
)


def test_get_oi_coverage_range_empty() -> None:
    assert get_oi_coverage_range([]) == (None, None)


def test_get_oi_coverage_range_min_max() -> None:
    assert get_oi_coverage_range([(3000, 1.0), (1000, 2.0), (2000, 3.0)]) == (
        1000,
        3000,
    )


def test_period_mapping_covers_hourly_bar() -> None:
    assert open_interest_period_for_candle_timeframe("1h") == "1h"


def test_period_mapping_finest_for_one_minute() -> None:
    assert open_interest_period_for_candle_timeframe("1m") == "5m"


def test_align_open_interest_backward_asof() -> None:
    ts = np.array([1000, 2000, 3000], dtype=np.int64)
    oi = [(500, 100.0), (1500, 110.0), (2500, 121.0)]
    out = align_open_interest_to_timestamps(ts, oi)
    np.testing.assert_allclose(out, [100.0, 110.0, 121.0])


def test_attach_open_interest_preserves_funding() -> None:
    c0 = OHLCV(
        symbol="BTC/USDT",
        timeframe="1h",
        timestamp=1000,
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.05,
        volume=1.0,
        funding_rate=0.0001,
    )
    out = attach_open_interest_to_ohlcv([c0], [(1000, 5000.0)])
    assert len(out) == 1
    assert out[0].open_interest == 5000.0
    assert out[0].funding_rate == 0.0001


def test_oi_change_series() -> None:
    ch = oi_change_series([100.0, 110.0, 99.0])
    assert np.isnan(ch[0])
    assert ch[1] == pytest.approx(0.1)
    # (99 - 110) / 110 ≈ -10%
    assert ch[2] == pytest.approx(-0.1)
