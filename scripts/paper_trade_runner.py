#!/usr/bin/env python3
"""Paper trading runner: periodic market fetch, rolling window, strategy replay, console logs.

FeatureEngine and MarketStateEngine are constructed inside BacktestEngine (same as production).
A new BacktestEngine is created each iteration so feature buffers do not leak across replays.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.disable(logging.INFO)

from alpha_engine.alpha.funding_asset_config import DEFAULT_FUNDING_CONFIRM_ASSET_CONFIG
from alpha_engine.alpha.funding_price_confirm_alpha import FundingPriceConfirmContrarianAlphaStrategy
from alpha_engine.backtest.engine import BacktestEngine
from alpha_engine.core.event_bus import EventBus
from alpha_engine.data.funding_rates import (
    attach_funding_to_ohlcv,
    fetch_funding_rate_history_usdm,
    timeframe_ms,
)
from alpha_engine.data.models import OHLCV
from alpha_engine.data_engine.binance_client import BinanceHistoricalClient

SYMBOL = "ETH/USDT"
TIMEFRAME = "1h"
SLEEP_SECONDS = 60  # run every minute
ROLLING_BARS = 500
POLL_BARS = 48  # refresh tail each loop after warm start (1h bars)
INITIAL_EQUITY = 100.0
Z_WINDOW = 168

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(
    LOG_DIR,
    f"paper_trade_{SYMBOL.replace('/', '_')}.log",
)


def log_output(text: str) -> None:
    print(text)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _periods_per_year_for_timeframe(timeframe: str) -> float:
    tf = timeframe.strip().lower()
    if tf.endswith("h"):
        return 365.0 * 24.0 / float(tf[:-1])
    if tf.endswith("m"):
        return 365.0 * 24.0 * 60.0 / float(tf[:-1])
    return 365.0 * 24.0 * 60.0


def _merge_trim(candles: list[OHLCV], batch: list[OHLCV]) -> list[OHLCV]:
    by_ts: dict[int, OHLCV] = {c.timestamp: c for c in candles}
    for c in batch:
        by_ts[c.timestamp] = c
    ordered = sorted(by_ts.values(), key=lambda c: c.timestamp)
    return ordered[-ROLLING_BARS:]


def _attach_funding(
    candles: list[OHLCV],
    *,
    symbol: str,
    timeframe: str,
    z_window: int,
) -> list[OHLCV]:
    if not candles:
        return []
    ts_sorted = sorted(c.timestamp for c in candles)
    t0, t1 = ts_sorted[0], ts_sorted[-1]
    warm_ms = max(
        14 * 86400 * 1000,
        (z_window + 50) * timeframe_ms(timeframe),
    )
    ev = fetch_funding_rate_history_usdm(symbol, t0 - warm_ms, t1)
    return attach_funding_to_ohlcv(candles, ev)


def _make_engine(event_bus: EventBus) -> BacktestEngine:
    strategy = FundingPriceConfirmContrarianAlphaStrategy(
        z_window=Z_WINDOW,
        asset_config=DEFAULT_FUNDING_CONFIRM_ASSET_CONFIG,
    )
    return BacktestEngine(
        event_bus=event_bus,
        initial_equity=INITIAL_EQUITY,
        strategies=[strategy],
        signal_filter=None,
        diagnostics=False,
        alpha_visibility=False,
        vol_regime_filter=False,
        simple_signal_execution=True,
        min_holding_period=0,
        cooldown_candles=0,
        min_steps_between_trades=0,
        periods_per_year=_periods_per_year_for_timeframe(TIMEFRAME),
        min_signal_threshold=0.0,
    )


def main() -> None:
    session_header = f"""
===============================
Paper Trading Session Started
Time: {datetime.utcnow()}
Symbol: {SYMBOL}
Timeframe: {TIMEFRAME}
===============================
"""

    log_output(session_header)

    client = BinanceHistoricalClient()
    candles: list[OHLCV] = []

    while True:
        try:
            if len(candles) < ROLLING_BARS:
                batch = client.fetch_ohlcv(
                    symbol=SYMBOL,
                    timeframe=TIMEFRAME,
                    limit=ROLLING_BARS,
                )
            else:
                batch = client.fetch_ohlcv(
                    symbol=SYMBOL,
                    timeframe=TIMEFRAME,
                    limit=min(POLL_BARS, ROLLING_BARS),
                )
            if not batch:
                log_output(f"[{datetime.utcnow()}] No candles returned; retrying.")
                time.sleep(5)
                continue

            candles = _merge_trim(candles, batch)
            funded = _attach_funding(
                candles,
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                z_window=Z_WINDOW,
            )

            bus = EventBus()
            last_net: list[float] = [0.0]

            def _on_net(sig: float) -> None:
                last_net[0] = float(sig)

            bus.subscribe("alpha.net_signal", _on_net)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                engine = _make_engine(bus)
                result = engine.run(funded)

            metrics = result.metrics
            current_position = engine.get_latest_position()
            last_price = funded[-1].close
            net_ret = float(metrics.get("net_total_return", 0.0))
            pnl_value = INITIAL_EQUITY * net_ret
            pnl_pct = net_ret * 100.0
            net_signal = last_net[0]

            output = f"""
[{datetime.utcnow()}]
Price: {last_price}
Position: {current_position:.4f}
Signal: net={net_signal:.6f}
PnL: {pnl_pct:.2f}% (${pnl_value:+.2f} vs ${INITIAL_EQUITY:.2f} start)
"""

            log_output(output)
            log_output("-" * 50)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            log_output(f"[ERROR] {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
