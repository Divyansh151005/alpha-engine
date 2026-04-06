#!/usr/bin/env python3
"""
Backtest validation: run with overtrading filters and print turnover, cost impact, trade count.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reduce log noise
logging.getLogger("alpha_engine").setLevel(logging.WARNING)

from alpha_engine.backtest import BacktestEngine
from alpha_engine.data.models import OHLCV


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


def _uptrend_candles(n: int, start: float = 100.0, step: float = 0.5) -> list[OHLCV]:
    return [_make_candle(i, start + i * step) for i in range(n)]


def main() -> None:
    candles = _uptrend_candles(300, start=100.0, step=0.3)

    engine = BacktestEngine(
        initial_equity=100.0,
        diagnostics=False,
        alpha_visibility=False,
    )
    result = engine.run(candles)

    print("=" * 60)
    print("BACKTEST VALIDATION (with overtrading filters)")
    print("=" * 60)
    print(f"  turnover:        {result.metrics.get('turnover', 0):.4f}")
    print(f"  total_cost:      {result.metrics.get('total_cost', 0):.4f}")
    print(f"  cost_impact:     {result.metrics.get('cost_impact', 0):.4f}")
    print(f"  trade_count:     {int(result.metrics.get('trade_count', 0))}")
    print(f"  total_return:    {result.metrics.get('total_return', 0):.4f}")
    print(f"  sharpe_ratio:    {result.metrics.get('sharpe_ratio', 0):.4f}")
    print(f"  max_drawdown:    {result.metrics.get('max_drawdown', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
