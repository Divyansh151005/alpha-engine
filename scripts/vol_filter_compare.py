#!/usr/bin/env python3
"""
Compare backtest metrics WITH vs WITHOUT execution vol regime filter (same data, same alpha).

Does not modify strategies — only scales target_position in HIGH_VOL (see BacktestEngine).
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("alpha_engine").setLevel(logging.WARNING)

from dotenv import load_dotenv

from alpha_engine.backtest import BacktestEngine
from alpha_engine.data.models import OHLCV
from alpha_engine.data_engine.binance_client import BinanceHistoricalClient


def _run(
    candles: list[OHLCV],
    *,
    vol_filter: bool,
) -> dict[str, float]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        engine = BacktestEngine(
            initial_equity=100.0,
            diagnostics=False,
            force_signal_debug=False,
            force_full_exposure=True,
            min_vol_scale=0.5,
            alpha_visibility=False,
            vol_regime_filter=vol_filter,
            vol_regime_signal="volume_spike",
            vol_regime_high_threshold=1.5,
        )
        result = engine.run(candles)
    return dict(result.metrics)


def main() -> None:
    load_dotenv()
    client = BinanceHistoricalClient()
    candles = client.fetch_ohlcv(symbol="BTC/USDT", timeframe="1m", limit=1000)
    if len(candles) < 200:
        print("Not enough candles.")
        sys.exit(1)

    m_off = _run(candles, vol_filter=False)
    m_on = _run(candles, vol_filter=True)

    def _m(d: dict[str, float], k: str) -> float:
        return float(d.get(k, 0.0))

    print("=" * 72)
print("VOL REGIME FILTER — same candles, same alpha stack")
print("Rank gate: |net| in top 10% vs last 50 bars  ->  else flat")
print("HIGH_VOL: volume_spike > 1.5  ->  target_position = 0 (hard gate)")
    print("=" * 72)
    print(f"{'metric':<22} {'WITHOUT filter':>18} {'WITH filter':>18}")
    print("-" * 72)
    for key in (
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "trade_count",
        "avg_trade_return",
    ):
        print(f"{key:<22} {_m(m_off, key):18.6f} {_m(m_on, key):18.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
